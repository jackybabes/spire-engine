//! A full run. Ports the run-level flow of `AbstractDungeon` (including moving between acts),
//! `AbstractRoom` rewards, campfires, chests, the boss chest and Neow.

pub mod acts;
pub mod map;
mod neow;
pub mod observe;
pub mod pools;
pub mod events;
mod relics;
pub mod shop;
#[cfg(test)]
mod event_tests;
#[cfg(test)]
mod relic_tests;

use serde::{Deserialize, Serialize};
use sts_rng::{java_shuffle, JavaRandom, StsRandom};

use crate::cards::{Card, CardId, CardRarity};
use crate::combat::{Combat, Outcome, PlayerAction};
use crate::encounters;
use crate::potions::PotionId;
use crate::relics::{RelicId, RelicState, RelicTier};
use crate::rngs::Rngs;
use acts::{act_data, ActData};
use map::{Map, RoomType};
pub use neow::{normalize_label, NeowDrawback, NeowOption, NeowReward};
use pools::{Pools, RewardRoom, SpawnCtx};
pub use crate::character::Character;

/// How a run starts. There is no profile progression: every card and relic is unlocked, every
/// ascension (0-20) is playable, Neow gives the full four blessings and every boss counts as
/// seen. The flags below only exist to reproduce a fresh profile for testing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub seed: i64,
    pub ascension: i32,
    /// Neow offers four blessings (a boss has been beaten on this profile) rather than two.
    pub neow_full: bool,
    /// Every boss has been seen before, so each act's boss is shuffled rather than forced.
    pub bosses_seen: bool,
    /// The final act is available: the three keys can be found and, with all three, the Act 3
    /// boss leads to Act 4. On by default (no progression); off reproduces a profile that has
    /// not won with every character.
    pub final_act: bool,
    #[serde(default)]
    pub character: Character,
    /// The keys are left to a hand-written rule instead of the player (`sts_encode::encode_run`), for a
    /// network that trained holding them (the Python module's `set_free_keys`). Off in the game and in the viewer.
    #[serde(default)]
    pub key_rule: Option<KeyRule>,
}

/// `RunConfig::key_rule`'s settings: from what percent of its HP a run takes Recall at a rest in Act 2 and in Act 3
/// (over 100: not in that act).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyRule {
    pub recall_hp_act2: i32,
    pub recall_hp_act3: i32,
}

impl RunConfig {
    pub fn new(seed: i64) -> RunConfig {
        RunConfig { seed, ascension: 0, neow_full: true, bosses_seen: true, final_act: true, character: Character::Ironclad, key_rule: None }
    }
}

/// One entry on the combat reward screen.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Reward {
    Gold(i32),
    StolenGold(i32),
    Relic(RelicId),
    Potion(PotionId),
    Cards(Vec<Card>),
    /// From the burning elite.
    EmeraldKey,
    /// From a chest, instead of the relic listed just before it.
    SapphireKey,
}

/// The keys to Act 4 (`Settings.hasRubyKey` and friends).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Keys {
    pub ruby: bool,
    pub emerald: bool,
    pub sapphire: bool,
}

impl Keys {
    pub fn all(&self) -> bool {
        self.ruby && self.emerald && self.sapphire
    }
}

/// What a grid selection of master-deck cards is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GridPurpose {
    Upgrade,
    Remove,
    Transform,
    Smith,
    /// Astrolabe: transform and upgrade.
    Astrolabe,
    /// A Bottled relic picks the card it holds.
    Bottle(RelicId),
    /// Peace Pipe at a rest site: remove a card (may be skipped).
    Toke,
    /// The merchant's card removal service.
    ShopPurge,
    /// An event's transform (miscRng, no upgrade).
    EventTransform,
    /// Bonfire Spirits: offer a card.
    Bonfire,
    /// Duplicator: obtain a copy of a card.
    Duplicate,
    /// Designer's full service: remove a card, then upgrade a random one.
    RemoveAndUpgrade,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Screen {
    Neow { options: Vec<NeowOption> },
    Map,
    Combat,
    Rewards { items: Vec<Reward> },
    /// Choosing a card from reward `item` (or from Neow when `item` is None).
    CardReward { item: Option<usize>, cards: Vec<Card> },
    Grid { purpose: GridPurpose, options: Vec<usize>, count: usize },
    Rest,
    BossRelic { relics: Vec<RelicId> },
    Shop { shop: Box<shop::Shop> },
    Event { event: Box<events::Event> },
    /// An event the port does not cover yet; the only action is to leave.
    Unsupported { what: String },
    Over,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunAction {
    Neow { option: usize },
    Path { x: i32 },
    Combat { action: PlayerAction },
    TakeReward { index: usize },
    PickCard { index: usize },
    SkipCards,
    Select { indices: Vec<usize> },
    Rest,
    Smith,
    /// Girya at a rest site.
    Lift,
    /// Peace Pipe at a rest site.
    Toke,
    /// Shovel at a rest site.
    Dig,
    /// Take the ruby key at a rest site.
    Recall,
    /// Singing Bowl on a card reward: +2 max HP instead of a card.
    SingingBowl,
    /// Drink a potion outside combat (Blood Potion, Fruit Juice, Entropic Brew).
    UsePotion { slot: usize },
    /// Shop: cards are indexed colored 0-4, then colorless 5-6.
    BuyCard { index: usize },
    BuyRelic { index: usize },
    BuyPotion { index: usize },
    /// Shop: pay to remove a card.
    Purge,
    /// Pick option `index` of the current event.
    EventChoice { index: usize },
    DiscardPotion { slot: usize },
    BossRelic { index: usize },
    Proceed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunOutcome {
    Died,
    /// Beat the last boss.
    Won,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Run {
    pub config: RunConfig,
    pub rng: Rngs,
    pub act: i32,
    pub floor: i32,
    pub hp: i32,
    pub max_hp: i32,
    pub gold: i32,
    pub relics: Vec<RelicState>,
    pub potions: Vec<Option<PotionId>>,
    pub deck: Vec<Card>,
    pub next_uuid: u32,
    pub next_instance: u32,
    pub map: Map,
    /// Current map position; None before the first room, `(x, 15)` at the boss.
    pub pos: Option<(i32, i32)>,
    pub room: Option<RoomType>,
    pub monster_list: Vec<String>,
    pub elite_list: Vec<String>,
    pub boss_list: Vec<String>,
    pub boss: String,
    pub pools: Pools,
    pub event_list: Vec<String>,
    pub shrine_list: Vec<String>,
    pub special_events: Vec<String>,
    /// EventHelper chances: elite, monster, shop, treasure.
    pub event_chances: [f32; 4],
    pub card_blizz: i32,
    pub potion_blizz: i32,
    pub combat: Option<Box<Combat>>,
    pub screen: Screen,
    pub outcome: Option<RunOutcome>,
    /// Encounter key of the current/last combat.
    pub encounter: Option<String>,
    /// Neow's card reward rolls happen with this stream (`NeowEvent.rng`).
    pub neow_rng: Option<StsRandom>,
    /// The reward list to return to after a card reward screen.
    saved_rewards: Option<Vec<Reward>>,
    /// Screens to return to when a sub-screen (a relic's grid or rewards) closes.
    #[serde(default)]
    screen_stack: Vec<Screen>,
    /// The reward screen up was opened with a label (`CombatRewardScreen.open(label)`: Cauldron,
    /// Orrery, Tiny House). The game hides Proceed there and leaves by the cancel button.
    #[serde(default)]
    labelled_rewards: bool,
    /// What was left on a labelled reward screen (a Cauldron, Orrery or Tiny House bought in this
    /// shop) when the player went back to the shop. `combatRewardScreen` keeps it, and the game
    /// reopens it when a grid closes in the shop (`closeCurrentScreen`, case GRID).
    #[serde(default)]
    shop_leftover_rewards: Vec<Reward>,
    /// `ShopScreen.purgeCost`: 75, then +25 per removal this run.
    #[serde(default = "default_purge_cost")]
    pub purge_cost: i32,
    /// Rewards an event added before its fight (Dead Adventurer, Mushrooms).
    #[serde(default)]
    event_rewards_pending: Option<Vec<Reward>>,
    /// The event fight sets `eliteTrigger` (Dead Adventurer).
    #[serde(default)]
    event_elite: bool,
    /// An event that comes back after its fight instead of showing rewards (Colosseum).
    #[serde(default)]
    event_resume: Option<Box<events::Event>>,
    /// Monsters an event made before its fight, and their encounter (`roll_event_monsters`).
    #[serde(default)]
    event_monsters: Option<(String, Vec<crate::monsters::Monster>)>,
    /// The Watcher's stance after the last fight: it lasts until the next room (`nextRoomTransition`
    /// resets it), so the Colosseum's second fight starts in it.
    #[serde(default)]
    stance: crate::combat::Stance,
    #[serde(default)]
    pub keys: Keys,
}

fn default_purge_cost() -> i32 {
    75
}

#[derive(Clone)]
struct MonsterInfo {
    name: &'static str,
    weight: f32,
}

fn normalize(weights: &[(&'static str, f32)]) -> Vec<MonsterInfo> {
    let mut list: Vec<MonsterInfo> = weights.iter().map(|&(name, weight)| MonsterInfo { name, weight }).collect();
    // Collections.sort by weight (stable), then divide by the float total.
    list.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap());
    let total: f32 = list.iter().fold(0.0f32, |acc, m| acc + m.weight);
    for m in list.iter_mut() {
        m.weight /= total;
    }
    list
}

fn roll_monster(list: &[MonsterInfo], roll: f32) -> String {
    let mut cur = 0.0f32;
    for m in list {
        cur += m.weight;
        if roll < cur {
            return m.name.to_string();
        }
    }
    "ERROR".to_string()
}

impl Run {
    /// A new run: `generateSeeds`, character creation, then the `Exordium` constructor.
    pub fn new(config: RunConfig) -> Run {
        let seed = config.seed;
        let rng = Rngs::new(seed);
        let character = config.character;
        let (common, uncommon, rare, colorless) = Pools::new_cards(character);
        let mut run = Run {
            rng: rng.clone(),
            act: 0,
            floor: 0,
            hp: character.max_hp(),
            max_hp: character.max_hp(),
            gold: 99,
            relics: vec![RelicState::new(character.starting_relic())],
            potions: vec![None; 3],
            deck: Vec::new(),
            next_uuid: 0,
            next_instance: 0,
            map: Map { rows: Vec::new() },
            pos: None,
            room: None,
            monster_list: Vec::new(),
            elite_list: Vec::new(),
            boss_list: Vec::new(),
            boss: String::new(),
            pools: Pools {
                common,
                uncommon,
                rare,
                colorless,
                common_relics: Vec::new(),
                uncommon_relics: Vec::new(),
                rare_relics: Vec::new(),
                shop_relics: Vec::new(),
                boss_relics: Vec::new(),
            },
            event_list: Vec::new(),
            shrine_list: Vec::new(),
            special_events: Vec::new(),
            event_chances: [0.0, 0.1, 0.03, 0.02],
            card_blizz: 5,
            potion_blizz: 0,
            combat: None,
            screen: Screen::Map,
            outcome: None,
            encounter: None,
            neow_rng: None,
            saved_rewards: None,
            screen_stack: Vec::new(),
            labelled_rewards: false,
            shop_leftover_rewards: Vec::new(),
            purge_cost: 75,
            event_rewards_pending: None,
            event_elite: false,
            event_resume: None,
            event_monsters: None,
            stance: crate::combat::Stance::Neutral,
            keys: Keys::default(),
            config,
        };
        // No profile progression: any ascension 0-20 can be played.
        run.config.ascension = run.config.ascension.clamp(0, 20);
        if run.config.ascension >= 14 {
            run.max_hp -= character.ascension_max_hp_loss();
        }
        // AbstractPlayer constructor: one fewer potion slot from A11.
        if run.config.ascension >= 11 {
            run.potions.pop();
        }
        // dungeonTransitionSetup
        run.act = 1;
        run.hp = run.max_hp;
        if run.config.ascension >= 6 {
            run.hp = map::java_round(run.max_hp as f32 * 0.9);
        }
        run.rng = rng;
        run.generate_act_content();
        // dungeonTransitionSetup adds Ascender's Bane before initializeStarterDeck, so it is the
        // deck's first card (the deck's order feeds every shuffle).
        if run.config.ascension >= 10 {
            let c = run.new_card(CardId::AscendersBane);
            run.deck.push(c);
        }
        // Starter deck.
        for id in character.starting_deck() {
            let c = run.new_card(id);
            run.deck.push(c);
        }
        // Exordium: relic pools, special events, map.
        let mut pools = Pools::new(&mut run.rng.relic, &[], character);
        std::mem::swap(&mut pools.common, &mut run.pools.common);
        std::mem::swap(&mut pools.uncommon, &mut run.pools.uncommon);
        std::mem::swap(&mut pools.rare, &mut run.pools.rare);
        std::mem::swap(&mut pools.colorless, &mut run.pools.colorless);
        run.pools = pools;
        run.special_events = [
            "Accursed Blacksmith",
            "Bonfire Elementals",
            "Designer",
            "Duplicator",
            "FaceTrader",
            "Fountain of Cleansing",
            "Knowing Skull",
            "Lab",
            "N'loth",
            "NoteForYourself",
            "SecretPortal",
            "The Joust",
            "WeMeetAgain",
            "The Woman in Blue",
        ]
        .iter()
        // `isNoteForYourselfAvailable`: at A0, or below A15 when the ascension is under the
        // character's highest unlocked one (profile ASCENSION_LEVEL). Every ascension is unlocked
        // here (highest 20), so that is every ascension below 15. The harness profile must say so
        // too (STSDataVagabond ASCENSION_LEVEL 20): without it the game leaves Note For Yourself
        // out at A1-14 and every later event roll differs.
        .filter(|s| **s != "NoteForYourself" || run.config.ascension < 15)
        .map(|s| s.to_string())
        .collect();
        run.generate_map();
        run.screen = neow::open(&mut run);
        run
    }

    /// This act's tables.
    pub fn act_data(&self) -> &'static ActData {
        act_data(self.act)
    }

    /// The `AbstractDungeon` constructor after `dungeonTransitionSetup`: encounter lists,
    /// bosses, events, shrines and card pools for the current act.
    fn generate_act_content(&mut self) {
        let d = self.act_data();
        if d.id == "TheEnding" {
            // Fixed lists; no monsterRng use.
            self.monster_list = vec!["Shield and Spear".to_string(); 3];
            self.elite_list = vec!["Shield and Spear".to_string(); 3];
            self.boss_list = vec!["The Heart".to_string(); 3];
            self.boss = self.boss_list[0].clone();
            self.event_list.clear();
            self.shrine_list.clear();
            let (common, uncommon, rare, colorless) = Pools::new_cards(self.config.character);
            self.pools.common = common;
            self.pools.uncommon = uncommon;
            self.pools.rare = rare;
            self.pools.colorless = colorless;
            return;
        }
        let mut r = self.rng.monster.clone();
        self.generate_weak(&mut r, d.weak_count);
        self.generate_strong(&mut r, 12);
        self.generate_elites(&mut r, 10);
        // initializeBoss
        if self.config.bosses_seen {
            self.boss_list = d.bosses.iter().map(|s| s.to_string()).collect();
            let s = r.random_long();
            java_shuffle(&mut self.boss_list, &mut JavaRandom::new(s));
        } else {
            self.boss_list = vec![d.first_boss.to_string(), d.first_boss.to_string()];
        }
        self.rng.monster = r;
        self.boss = self.boss_list[0].clone();
        self.event_list = d.events.iter().map(|s| s.to_string()).collect();
        self.shrine_list = d.shrines.iter().map(|s| s.to_string()).collect();
        let (common, uncommon, rare, colorless) = Pools::new_cards(self.config.character);
        self.pools.common = common;
        self.pools.uncommon = uncommon;
        self.pools.rare = rare;
        self.pools.colorless = colorless;
    }

    /// `mapRng = new Random(seed + actNum * k)`, then `generateMap()`.
    fn generate_map(&mut self) {
        let d = self.act_data();
        self.rng.map = StsRandom::new(self.config.seed.wrapping_add(self.act as i64 * d.map_seed_mult));
        if d.id == "TheEnding" {
            self.map = Map::the_ending();
            return;
        }
        let emerald = self.config.final_act && !self.keys.emerald;
        self.map = Map::generate(&mut self.rng.map, &d.chances, self.config.ascension, emerald);
    }

    /// Leaving the boss chest (`ProceedButton.goToNextDungeon`): `dungeonTransitionSetup` and
    /// the next act's constructor.
    fn next_act(&mut self) {
        self.act += 1;
        let c = self.rng.card.counter;
        if c > 0 && c < 250 {
            self.rng.card.set_counter(250);
        } else if c > 250 && c < 500 {
            self.rng.card.set_counter(500);
        } else if c > 500 && c < 750 {
            self.rng.card.set_counter(750);
        }
        // EventHelper.resetProbabilities
        self.event_chances = [0.0, 0.1, 0.03, 0.02];
        self.event_list.clear();
        self.shrine_list.clear();
        self.monster_list.clear();
        self.elite_list.clear();
        self.boss_list.clear();
        self.potion_blizz = 0;
        if self.config.ascension >= 5 {
            self.heal(map::java_round((self.max_hp - self.hp) as f32 * 0.75));
        } else {
            self.heal(self.max_hp);
        }
        self.generate_act_content();
        self.generate_map();
        // currMapNode = new MapRoomNode(0, -1) with an EmptyRoom.
        self.pos = None;
        self.room = None;
        self.screen = Screen::Map;
    }

    fn generate_weak(&mut self, rng: &mut StsRandom, count: usize) {
        let list = normalize(self.act_data().weak);
        self.populate(rng, &list, count, false);
    }

    fn generate_strong(&mut self, rng: &mut StsRandom, count: usize) {
        let d = self.act_data();
        let list = normalize(d.strong);
        // populateFirstStrongEnemy with generateExclusions
        let exclusions = self.monster_list.last().map_or(&[][..], |m| (d.exclusions)(m));
        loop {
            let m = roll_monster(&list, rng.random_float());
            if !exclusions.contains(&m.as_str()) {
                self.monster_list.push(m);
                break;
            }
        }
        self.populate(rng, &list, count, false);
    }

    fn generate_elites(&mut self, rng: &mut StsRandom, count: usize) {
        let list = normalize(self.act_data().elites);
        self.populate(rng, &list, count, true);
    }

    /// `populateMonsterList`.
    fn populate(&mut self, rng: &mut StsRandom, list: &[MonsterInfo], count: usize, elites: bool) {
        let mut i = 0;
        while i < count {
            let target = if elites { &mut self.elite_list } else { &mut self.monster_list };
            let m = roll_monster(list, rng.random_float());
            if target.is_empty() {
                target.push(m);
            } else if m != *target.last().unwrap() {
                if !elites && target.len() > 1 && m == target[target.len() - 2] {
                    continue;
                }
                target.push(m);
            } else {
                continue;
            }
            i += 1;
        }
    }

    pub fn new_card(&mut self, id: CardId) -> Card {
        let c = Card::new(id, self.next_uuid, self.next_instance);
        self.next_uuid += 1;
        self.next_instance += 1;
        c
    }

    pub fn has_relic(&self, id: RelicId) -> bool {
        self.relics.iter().any(|r| r.id == id)
    }

    /// `returnRandomRelic(tier)`.
    pub fn random_relic(&mut self, tier: RelicTier) -> RelicId {
        let relics = self.relics.clone();
        let deck = self.deck.clone();
        let ctx = SpawnCtx { floor: self.floor, act: self.act, in_shop: self.room == Some(RoomType::Shop), relics: &relics, deck: &deck };
        self.pools.take_relic(tier, &ctx)
    }

    /// `returnRandomNonCampfireRelic(tier)`.
    pub fn random_noncamp_relic(&mut self, tier: RelicTier) -> RelicId {
        loop {
            let r = self.random_relic(tier);
            if !matches!(r, RelicId::PeacePipe | RelicId::Shovel | RelicId::Girya) {
                return r;
            }
        }
    }

    pub fn gain_gold(&mut self, amount: i32) {
        if self.has_relic(RelicId::Ectoplasm) || amount <= 0 {
            return;
        }
        self.gold += amount;
        // onGainGold: Bloody Idol.
        if self.has_relic(RelicId::BloodyIdol) {
            self.heal(5);
        }
    }

    /// `obtainPotion`: first empty slot, or false when full.
    pub fn obtain_potion(&mut self, p: PotionId) -> bool {
        if let Some(slot) = self.potions.iter_mut().find(|s| s.is_none()) {
            *slot = Some(p);
            true
        } else {
            false
        }
    }

    /// `AbstractCreature.heal` outside combat (Mark of the Bloom's `onPlayerHeal`).
    pub fn heal(&mut self, amount: i32) {
        if self.has_relic(RelicId::MarkOfTheBloom) {
            return;
        }
        self.hp = (self.hp + amount).min(self.max_hp);
    }

    pub fn increase_max_hp(&mut self, amount: i32) {
        self.max_hp += amount;
        self.heal(amount);
    }

    pub fn decrease_max_hp(&mut self, amount: i32) {
        self.max_hp = (self.max_hp - amount).max(1);
        self.hp = self.hp.min(self.max_hp);
    }

    // ---- Card rewards ----

    /// `rollRarity()` with the room's chances (`AbstractRoom.getCardRarity`).
    fn roll_card_rarity(&mut self, room: RewardRoom) -> CardRarity {
        let roll = self.rng.card.random(99) + self.card_blizz;
        let (mut rare, uncommon) = match room {
            RewardRoom::Boss => return CardRarity::Rare,
            RewardRoom::Elite => (10, 40),
            RewardRoom::Shop => (9, 37),
            _ => (3, 37),
        };
        // alterCardRarityProbabilities (not in shops or rest sites): N'loth's Gift.
        if matches!(room, RewardRoom::Normal | RewardRoom::Elite) && self.has_relic(RelicId::NlothsGift) {
            rare *= 3;
        }
        if roll < rare {
            CardRarity::Rare
        } else if roll >= rare + uncommon {
            CardRarity::Common
        } else {
            CardRarity::Uncommon
        }
    }

    /// `AbstractDungeon.getRewardCards()`.
    pub fn reward_cards(&mut self, room: RewardRoom) -> Vec<Card> {
        let mut n = 3;
        if self.has_relic(RelicId::QuestionCard) {
            n += 1;
        }
        if self.has_relic(RelicId::BustedCrown) {
            n -= 2;
        }
        let mut ids: Vec<CardId> = Vec::new();
        for _ in 0..n {
            let rarity = self.roll_card_rarity(room);
            match rarity {
                CardRarity::Common => {
                    self.card_blizz -= 1;
                    if self.card_blizz <= -40 {
                        self.card_blizz = -40;
                    }
                }
                CardRarity::Rare => self.card_blizz = 5,
                _ => {}
            }
            let card = loop {
                let c = if self.has_relic(RelicId::PrismaticShard) {
                    pools::Pools::any_color_card(rarity, &mut self.rng.card)
                } else {
                    self.pools.random_card(rarity, &mut self.rng.card)
                };
                if !ids.contains(&c) {
                    break c;
                }
            };
            ids.push(card);
        }
        let mut out = Vec::new();
        for id in ids {
            let mut c = self.new_card(id);
            let chance = self.act_data().card_upgraded_chance(self.config.ascension);
            if c.data().rarity != CardRarity::Rare && self.rng.card.random_boolean_chance(chance) && c.can_upgrade() {
                c.upgrade();
            } else {
                self.preview_obtain(&mut c);
            }
            out.push(c);
        }
        out
    }

    /// `AbstractRoom.addPotionToRewards()`.
    fn add_potion_reward(&mut self, items: &mut Vec<Reward>, eligible: bool) {
        let mut chance = if eligible { 40 + self.potion_blizz } else { 0 };
        if self.has_relic(RelicId::WhiteBeast) {
            chance = 100;
        }
        if items.len() >= 4 {
            chance = 0;
        }
        if self.rng.potion.random_range(0, 99) >= chance {
            self.potion_blizz += 10;
        } else {
            let p = pools::return_random_potion(self.config.character, &mut self.rng.potion);
            items.push(Reward::Potion(p));
            self.potion_blizz -= 10;
        }
    }

    fn add_gold(items: &mut Vec<Reward>, gold: i32) {
        for i in items.iter_mut() {
            if let Reward::Gold(g) = i {
                *g += gold;
                return;
            }
        }
        items.push(Reward::Gold(gold));
    }

    // ---- Agent interface ----

    pub fn is_over(&self) -> bool {
        self.outcome.is_some()
    }

    /// Choices available on the current screen.
    pub fn legal_actions(&self) -> Vec<RunAction> {
        let mut out = self.legal_actions_screen();
        if matches!(self.screen, Screen::Map | Screen::Rewards { .. } | Screen::Rest | Screen::BossRelic { .. } | Screen::Shop { .. } | Screen::Unsupported { .. }) {
            for (slot, p) in self.potions.iter().enumerate() {
                let Some(p) = p else { continue };
                if matches!(p, PotionId::BloodPotion | PotionId::FruitJuice | PotionId::EntropicBrew) {
                    out.push(RunAction::UsePotion { slot });
                }
                out.push(RunAction::DiscardPotion { slot });
            }
        }
        out
    }

    fn legal_actions_screen(&self) -> Vec<RunAction> {
        let mut out = Vec::new();
        match &self.screen {
            Screen::Neow { options } => {
                for i in 0..options.len() {
                    out.push(RunAction::Neow { option: i });
                }
            }
            Screen::Map => {
                for x in self.reachable_nodes() {
                    out.push(RunAction::Path { x });
                }
            }
            Screen::Combat => {
                if let Some(c) = &self.combat {
                    out.extend(c.legal_actions().into_iter().map(|a| RunAction::Combat { action: a }));
                }
            }
            Screen::Rewards { items } => {
                for (i, it) in items.iter().enumerate() {
                    let ok = match it {
                        Reward::Potion(_) => self.potions.iter().any(|p| p.is_none()) || self.has_relic(RelicId::Sozu),
                        _ => true,
                    };
                    if ok {
                        out.push(RunAction::TakeReward { index: i });
                    }
                }
                out.push(RunAction::Proceed);
            }
            Screen::CardReward { cards, .. } => {
                for i in 0..cards.len() {
                    out.push(RunAction::PickCard { index: i });
                }
                if self.has_relic(RelicId::SingingBowl) {
                    out.push(RunAction::SingingBowl);
                }
                out.push(RunAction::SkipCards);
            }
            Screen::Grid { options, count, .. } => {
                // No cancel on a grid, the Toke one included: the game's cancel there reopens the
                // campfire (`CancelButton`: `campfireUI.reopen`), where greedy play could toke and
                // cancel for ever. The engine's cancel used to leave the rest site instead (run H142
                // of the 2026-10-28 Watcher soak, stuck at the reopened campfire).
                fn combos(opts: &[usize], k: usize, start: usize, cur: &mut Vec<usize>, out: &mut Vec<RunAction>) {
                    if cur.len() == k {
                        out.push(RunAction::Select { indices: cur.clone() });
                        return;
                    }
                    for i in start..opts.len() {
                        cur.push(opts[i]);
                        combos(opts, k, i + 1, cur, out);
                        cur.pop();
                    }
                }
                combos(options, (*count).min(options.len()), 0, &mut Vec::new(), &mut out);
            }
            Screen::Rest => {
                out.extend(self.campfire_options());
                if out.is_empty() {
                    out.push(RunAction::Proceed);
                }
            }
            Screen::BossRelic { relics } => {
                for i in 0..relics.len() {
                    out.push(RunAction::BossRelic { index: i });
                }
                out.push(RunAction::Proceed);
            }
            Screen::Shop { shop } => {
                for (i, c) in shop.cards.iter().enumerate() {
                    if c.as_ref().is_some_and(|c| c.price <= self.gold) {
                        out.push(RunAction::BuyCard { index: i });
                    }
                }
                for (i, r) in shop.relics.iter().enumerate() {
                    if r.as_ref().is_some_and(|r| r.price <= self.gold) {
                        out.push(RunAction::BuyRelic { index: i });
                    }
                }
                let free_slot = self.potions.iter().any(|p| p.is_none());
                for (i, p) in shop.potions.iter().enumerate() {
                    if free_slot && !self.has_relic(RelicId::Sozu) && p.as_ref().is_some_and(|p| p.price <= self.gold) {
                        out.push(RunAction::BuyPotion { index: i });
                    }
                }
                if shop.purge_available && shop.purge_cost <= self.gold && !self.shop_purgeable().is_empty() {
                    out.push(RunAction::Purge);
                }
                out.push(RunAction::Proceed);
            }
            Screen::Event { event } => {
                for (i, &on) in event.options.iter().enumerate() {
                    if on {
                        out.push(RunAction::EventChoice { index: i });
                    }
                }
            }
            Screen::Unsupported { .. } => out.push(RunAction::Proceed),
            Screen::Over => {}
        }
        out
    }

    /// Map x positions reachable next (row 0 at the start, the boss after row 14).
    pub fn next_nodes(&self) -> Vec<i32> {
        match self.pos {
            None => (0..map::MAP_WIDTH as i32).filter(|&x| !self.map.node(x, 0).edges.is_empty()).collect(),
            Some((_, y)) if y >= map::MAP_HEIGHT as i32 - 1 => vec![3],
            Some((x, y)) => self.map.node(x, y).edges.iter().filter(|e| e.1 == y + 1).map(|e| e.0).collect(),
        }
    }

    pub fn step(&mut self, action: &RunAction) -> Result<(), String> {
        if self.outcome.is_some() {
            return Err("run is over".into());
        }
        match action {
            RunAction::UsePotion { slot } => return self.use_potion_outside(*slot),
            RunAction::DiscardPotion { slot } if !matches!(self.screen, Screen::Combat) => {
                let s = self.potions.get_mut(*slot).ok_or("no such potion slot")?;
                return if s.take().is_some() { Ok(()) } else { Err("that slot is empty".into()) };
            }
            _ => {}
        }
        let screen = self.screen.clone();
        match (screen, action) {
            (Screen::Neow { options }, RunAction::Neow { option }) => {
                let opt = options.get(*option).ok_or("no such Neow option")?.clone();
                self.screen = Screen::Map;
                neow::choose(self, opt);
            }
            (Screen::Map, RunAction::Path { x }) => {
                if !self.reachable_nodes().contains(x) {
                    return Err("that node is not reachable".into());
                }
                if !self.next_nodes().contains(x) {
                    // Winged Greaves: flying to an unconnected node uses a charge.
                    if let Some(w) = self.relics.iter_mut().find(|r| r.id == RelicId::WingBoots) {
                        w.counter -= 1;
                        if w.counter <= 0 {
                            w.counter = -2;
                            w.used_up = true;
                        }
                    }
                }
                self.enter_node(*x);
            }
            (Screen::Combat, RunAction::Combat { action }) => {
                let c = self.combat.as_mut().ok_or("no combat")?;
                c.step(action)?;
                if c.is_over() {
                    self.finish_combat();
                }
            }
            (Screen::Rewards { mut items }, RunAction::TakeReward { index }) => {
                let it = items.get(*index).ok_or("no such reward")?.clone();
                match it {
                    Reward::Gold(g) => {
                        let total = self.gold_with_bonus(g);
                        self.gain_gold(total);
                        items.remove(*index);
                    }
                    Reward::StolenGold(g) => {
                        self.gain_gold(g);
                        items.remove(*index);
                    }
                    Reward::Relic(r) => {
                        items.remove(*index);
                        // A chest relic and the sapphire key after it exclude each other.
                        if items.get(*index) == Some(&Reward::SapphireKey) {
                            items.remove(*index);
                        }
                        self.screen = Screen::Rewards { items };
                        self.obtain_relic(r);
                        return Ok(());
                    }
                    Reward::EmeraldKey => {
                        self.keys.emerald = true;
                        items.remove(*index);
                    }
                    Reward::SapphireKey => {
                        self.keys.sapphire = true;
                        items.remove(*index);
                        if *index > 0 && matches!(items.get(*index - 1), Some(Reward::Relic(_))) {
                            items.remove(*index - 1);
                        }
                    }
                    Reward::Potion(p) => {
                        if self.has_relic(RelicId::Sozu) || self.obtain_potion(p) {
                            items.remove(*index);
                        } else {
                            return Err("potion slots are full".into());
                        }
                    }
                    Reward::Cards(cards) => {
                        self.saved_rewards = Some(items);
                        self.screen = Screen::CardReward { item: Some(*index), cards };
                        return Ok(());
                    }
                }
                self.screen = Screen::Rewards { items };
            }
            (Screen::Rewards { items }, RunAction::Proceed) => {
                if self.labelled_rewards && matches!(self.screen_stack.last(), Some(Screen::Shop { .. })) {
                    self.shop_leftover_rewards = items;
                }
                self.leave_rewards()
            }
            (Screen::CardReward { item, cards }, RunAction::PickCard { index }) => {
                let c = cards.get(*index).ok_or("no such card")?.clone();
                self.obtain_card(c);
                self.after_card_reward(item, true);
            }
            (Screen::CardReward { item, .. }, RunAction::SkipCards) => self.after_card_reward(item, false),
            (Screen::CardReward { item, .. }, RunAction::SingingBowl) if self.has_relic(RelicId::SingingBowl) => {
                self.increase_max_hp(2);
                self.after_card_reward(item, true);
            }
            (Screen::Grid { purpose, options, count }, RunAction::Select { indices }) => {
                if indices.len() != count || indices.iter().any(|i| !options.contains(i)) {
                    return Err(format!("select {count} card(s) from the options"));
                }
                self.resolve_grid(purpose, indices);
            }
            (Screen::Rest, a @ (RunAction::Rest | RunAction::Smith | RunAction::Lift | RunAction::Toke | RunAction::Dig | RunAction::Recall)) => {
                if !self.campfire_options().contains(a) {
                    return Err(format!("{a:?} is not available here"));
                }
                self.screen = Screen::Map;
                match a {
                    RunAction::Rest => {
                        let mut amt = (self.max_hp as f32 * 0.3) as i32;
                        if self.has_relic(RelicId::RegalPillow) {
                            amt += 15;
                        }
                        self.heal(amt);
                        if self.has_relic(RelicId::DreamCatcher) {
                            let cards = self.reward_cards(RewardRoom::Rest);
                            if !cards.is_empty() {
                                self.screen = Screen::CardReward { item: None, cards };
                            }
                        }
                    }
                    RunAction::Smith => {
                        let options: Vec<usize> = (0..self.deck.len()).filter(|&i| self.deck[i].can_upgrade()).collect();
                        self.screen = Screen::Grid { purpose: GridPurpose::Smith, options, count: 1 };
                    }
                    RunAction::Lift => {
                        if let Some(g) = self.relics.iter_mut().find(|r| r.id == RelicId::Girya) {
                            g.counter += 1;
                        }
                    }
                    RunAction::Recall => self.keys.ruby = true,
                    RunAction::Toke => {
                        let options: Vec<usize> = neow::purgeable(self).into_iter().filter(|&i| !self.deck[i].bottled).collect();
                        self.screen = Screen::Grid { purpose: GridPurpose::Toke, options, count: 1 };
                    }
                    _ => {
                        let tier = self.random_relic_tier();
                        let r = self.random_relic(tier);
                        self.screen = Screen::Rewards { items: vec![Reward::Relic(r)] };
                    }
                }
            }
            (Screen::Rest, RunAction::Proceed) => self.screen = Screen::Map,
            (Screen::BossRelic { relics }, RunAction::BossRelic { index }) => {
                let r = *relics.get(*index).ok_or("no such relic")?;
                self.screen = Screen::Over;
                self.obtain_relic(r);
                if matches!(self.screen, Screen::Over) {
                    self.finish_act();
                }
            }
            (Screen::BossRelic { .. }, RunAction::Proceed) => self.finish_act(),
            // An unported boss is skipped so the run can carry on to the next act.
            (Screen::Unsupported { .. }, RunAction::Proceed) if self.room == Some(RoomType::Boss) => self.enter_boss_chest(),
            (Screen::Unsupported { .. }, RunAction::Proceed) => self.screen = Screen::Map,
            (Screen::Shop { shop }, RunAction::BuyCard { index }) => self.buy_card(shop, *index)?,
            (Screen::Shop { shop }, RunAction::BuyRelic { index }) => self.buy_relic(shop, *index)?,
            (Screen::Shop { shop }, RunAction::BuyPotion { index }) => self.buy_potion(shop, *index)?,
            (Screen::Shop { shop }, RunAction::Purge) => self.start_purge(shop)?,
            (Screen::Shop { .. }, RunAction::Proceed) => {
                self.shop_leftover_rewards.clear();
                self.screen = Screen::Map
            }
            (Screen::Event { event }, RunAction::EventChoice { index }) => self.event_choose(*event, *index)?,
            (s, a) => return Err(format!("{a:?} is not valid on {}", screen_name(&s))),
        }
        Ok(())
    }

    /// Potions usable outside combat.
    fn use_potion_outside(&mut self, slot: usize) -> Result<(), String> {
        if !self.legal_actions().contains(&RunAction::UsePotion { slot }) {
            return Err("that potion cannot be used here".into());
        }
        let p = self.potions[slot].ok_or("empty slot")?;
        let n = p.data().potency * if self.has_relic(RelicId::SacredBark) { 2 } else { 1 };
        match p {
            PotionId::BloodPotion => {
                let amt = (self.max_hp as f32 * (n as f32 / 100.0)) as i32;
                self.heal(amt);
            }
            PotionId::FruitJuice => self.increase_max_hp(n),
            _ => {}
        }
        if self.has_relic(RelicId::ToyOrnithopter) {
            self.heal(5);
        }
        self.potions[slot] = None;
        if p == PotionId::EntropicBrew && !self.has_relic(RelicId::Sozu) {
            for _ in 0..self.potions.len() {
                let np = pools::return_random_potion(self.config.character, &mut self.rng.potion);
                self.obtain_potion(np);
            }
        }
        Ok(())
    }

    fn after_card_reward(&mut self, item: Option<usize>, picked: bool) {
        match item {
            Some(i) => {
                // Back to the reward list; a picked card reward is consumed.
                let mut items = self.saved_rewards.take().unwrap_or_default();
                if picked && i < items.len() {
                    items.remove(i);
                }
                self.screen = Screen::Rewards { items };
            }
            None => self.pop_screen(),
        }
    }

    /// The reward screen up leaves by the game's cancel button, not Proceed (`labelled_rewards`).
    pub fn rewards_labelled(&self) -> bool {
        self.labelled_rewards
    }

    fn leave_rewards(&mut self) {
        self.labelled_rewards = false;
        if !self.screen_stack.is_empty() {
            self.pop_screen();
        } else if self.room == Some(RoomType::Boss) {
            self.enter_boss_chest();
        } else {
            self.screen = Screen::Map;
        }
    }

    /// `CampfireUI` options that can be used (`addCampfireOption`, `canUseCampfireOption`).
    pub fn campfire_options(&self) -> Vec<RunAction> {
        let mut out = Vec::new();
        if !self.has_relic(RelicId::CoffeeDripper) {
            out.push(RunAction::Rest);
        }
        if !self.has_relic(RelicId::FusionHammer) && self.deck.iter().any(|c| c.can_upgrade()) {
            out.push(RunAction::Smith);
        }
        for r in &self.relics {
            match r.id {
                RelicId::Girya if r.counter < 3 => out.push(RunAction::Lift),
                RelicId::PeacePipe if neow::purgeable(self).iter().any(|&i| !self.deck[i].bottled) => out.push(RunAction::Toke),
                RelicId::Shovel => out.push(RunAction::Dig),
                _ => {}
            }
        }
        if self.config.final_act && !self.keys.ruby {
            out.push(RunAction::Recall);
        }
        out
    }

    /// Map moves: connected nodes, plus the whole next row while Winged Greaves has charges.
    pub fn reachable_nodes(&self) -> Vec<i32> {
        let mut out = self.next_nodes();
        let y = self.pos.map_or(0, |(_, y)| y + 1);
        let wings = self.relics.iter().any(|r| r.id == RelicId::WingBoots && r.counter > 0);
        if wings && self.pos.is_some() && y < map::MAP_HEIGHT as i32 {
            for x in 0..map::MAP_WIDTH as i32 {
                let n = self.map.node(x, y);
                if (!n.edges.is_empty() || !n.parents.is_empty()) && !out.contains(&x) {
                    out.push(x);
                }
            }
            out.sort_unstable();
        }
        out
    }

    /// A gold reward with `applyGoldBonus`: Golden Idol adds 25% outside treasure rooms. The boss
    /// chest (`TreasureRoomBoss`, not a `TreasureRoom`; the engine's room there is Treasure, on the
    /// boss row) counts as outside: Tiny House's 50 gold is 63 with Golden Idol (run H0XB of the
    /// 2026-10-28 Watcher soak).
    pub fn gold_with_bonus(&self, g: i32) -> i32 {
        let boss_chest = self.pos.is_some_and(|(_, y)| y == 15);
        if self.has_relic(RelicId::GoldenIdol) && (self.room != Some(RoomType::Treasure) || boss_chest) {
            g + map::java_round(g as f32 * 0.25)
        } else {
            g
        }
    }

    /// `ProceedButton` after an Act 3 boss: the second boss at A20, then the door (Act 4 when
    /// the keys are in hand) or the victory room.
    fn after_beyond_boss(&mut self) {
        if self.config.ascension >= 20 && self.boss_list.len() == 2 {
            // goToDoubleBoss: bossKey = bossList[0], a fresh boss room.
            self.boss = self.boss_list[0].clone();
            let x = self.pos.map_or(0, |(x, _)| x);
            self.pos = Some((x, map::MAP_HEIGHT as i32 - 1));
            self.enter_node(x);
            return;
        }
        // goToVictoryRoomOrTheDoor: the VictoryRoom (the Spire Heart event) is a room of its own,
        // entered by nextRoomTransition: a floor, the floor's RNG seeds, the relics' room hooks
        // (only Maw Bank's applies there). Without it the engine's Act 4 floors were one behind the
        // game's, and so its shuffles (playback of an A10 Heart win drew other hands, 2026-10-25).
        self.floor += 1;
        self.rng.enter_floor(self.floor);
        self.relics_on_enter_room(RoomType::Boss);
        if self.config.final_act && self.keys.all() {
            // The Spire Heart event opens the door to TheEnding.
            self.next_act();
            return;
        }
        self.outcome = Some(RunOutcome::Won);
        self.screen = Screen::Over;
    }

    /// Leaving the boss chest: on to the next act.
    fn finish_act(&mut self) {
        if self.act < 3 {
            self.next_act();
        } else {
            self.outcome = Some(RunOutcome::Won);
            self.screen = Screen::Over;
        }
    }

    fn resolve_grid(&mut self, purpose: GridPurpose, indices: &[usize]) {
        match purpose {
            GridPurpose::Upgrade | GridPurpose::Smith => self.deck[indices[0]].upgrade(),
            GridPurpose::Remove | GridPurpose::Toke => {
                self.take_cards(indices);
                self.on_master_deck_change();
            }
            GridPurpose::Transform => neow::transform(self, indices),
            GridPurpose::Astrolabe => self.astrolabe(indices),
            GridPurpose::ShopPurge => self.finish_purge(indices),
            GridPurpose::EventTransform => self.event_transform(indices),
            GridPurpose::Bonfire => {
                let card = self.deck[indices[0]].clone();
                self.bonfire_offer(&card);
                self.take_cards(indices);
                self.on_master_deck_change();
            }
            GridPurpose::Duplicate => {
                let src = self.deck[indices[0]].clone();
                let mut c = self.new_card(src.id);
                for _ in 0..src.times_upgraded {
                    c.upgrade();
                }
                c.misc = src.misc;
                c.base_damage = src.base_damage;
                self.obtain_card(c);
            }
            GridPurpose::RemoveAndUpgrade => {
                self.take_cards(indices);
                self.on_master_deck_change();
                let mut cands: Vec<usize> = (0..self.deck.len()).filter(|&i| self.deck[i].can_upgrade()).collect();
                let seed = self.rng.misc.random_long();
                java_shuffle(&mut cands, &mut JavaRandom::new(seed));
                if let Some(&i) = cands.first() {
                    self.deck[i].upgrade();
                }
            }
            GridPurpose::Bottle(id) => {
                let uuid = self.deck[indices[0]].uuid;
                self.deck[indices[0]].bottled = true;
                if let Some(r) = self.relics.iter_mut().find(|r| r.id == id) {
                    r.card_uuid = Some(uuid);
                }
            }
        }
        self.pop_screen();
        if matches!(self.screen, Screen::Shop { .. }) && !self.shop_leftover_rewards.is_empty() {
            // Closing a grid with rewards left reopens the reward screen, not the shop (soak run
            // KZPH: a shop removal after skipping some of a Cauldron's potions).
            let items = self.shop_leftover_rewards.clone();
            self.labelled_rewards = true;
            self.push_screen(Screen::Rewards { items });
        }
    }

    // ---- Rooms ----

    /// `nextRoomTransition` into map node x of the next row (or the boss).
    fn enter_node(&mut self, x: i32) {
        // Leaving the current room trims the encounter lists.
        match self.room {
            Some(RoomType::Elite) => {
                if !self.elite_list.is_empty() {
                    self.elite_list.remove(0);
                } else {
                    let mut r = self.rng.monster.clone();
                    self.generate_elites(&mut r, 10);
                    self.rng.monster = r;
                }
            }
            Some(RoomType::Monster) | Some(RoomType::Boss) => {
                if !self.monster_list.is_empty() {
                    self.monster_list.remove(0);
                } else {
                    let mut r = self.rng.monster.clone();
                    self.generate_strong(&mut r, 12);
                    self.rng.monster = r;
                }
            }
            _ => {}
        }
        self.floor += 1;
        self.rng.enter_floor(self.floor);
        self.event_monsters = None;
        self.stance = crate::combat::Stance::Neutral;
        let y = self.pos.map_or(0, |(_, y)| y + 1);
        let mut room = if y >= map::MAP_HEIGHT as i32 { RoomType::Boss } else { self.map.node(x, y).room.unwrap_or(RoomType::Monster) };
        self.relics_on_enter_room(room);
        if room == RoomType::Event {
            let mut dup = StsRandom::with_counter(self.config.seed, self.rng.event.counter);
            room = self.roll_event_room(&mut dup);
            self.rng.event = dup;
        }
        self.pos = Some((x, y));
        self.room = Some(room);
        self.relics_just_entered_room(room);
        match room {
            RoomType::Monster => {
                let key = if self.monster_list.is_empty() {
                    let mut r = self.rng.monster.clone();
                    self.generate_strong(&mut r, 12);
                    self.rng.monster = r;
                    self.monster_list[0].clone()
                } else {
                    self.monster_list[0].clone()
                };
                self.start_combat(&key);
            }
            RoomType::Elite => {
                if self.elite_list.is_empty() {
                    let mut r = self.rng.monster.clone();
                    self.generate_elites(&mut r, 10);
                    self.rng.monster = r;
                }
                let key = self.elite_list[0].clone();
                self.start_combat(&key);
            }
            RoomType::Boss => {
                let key = self.boss.clone();
                if !self.boss_list.is_empty() {
                    self.boss_list.remove(0);
                }
                self.start_combat(&key);
            }
            RoomType::Rest => {
                self.relics_on_enter_rest_room();
                self.screen = Screen::Rest;
            }
            RoomType::Treasure => self.open_chest(),
            RoomType::Shop => self.open_shop(),
            RoomType::Event => {
                let key = self.generate_event();
                match events::EventId::from_key(&key) {
                    Some(id) => self.open_event(id),
                    None => self.screen = Screen::Unsupported { what: format!("event {key}") },
                }
            }
        }
    }

    /// `EventHelper.roll(eventRng)`.
    fn roll_event_room(&mut self, rng: &mut StsRandom) -> RoomType {
        let roll = rng.random_float();
        let mut force_chest = false;
        if let Some(r) = self.relics.iter_mut().find(|r| r.id == RelicId::TinyChest) {
            r.counter += 1;
            if r.counter == 4 {
                r.counter = 0;
                force_chest = true;
            }
        }
        let [elite, monster, shop, treasure] = self.event_chances;
        let elite_size = 0;
        let monster_size = (monster * 100.0) as i32;
        let shop_size = if self.room == Some(RoomType::Shop) { 0 } else { (shop * 100.0) as i32 };
        let treasure_size = (treasure * 100.0) as i32;
        let mut results = [RoomType::Event; 100];
        let mut fill = 0i32;
        let fill_range = |from: i32, n: i32, r: RoomType, results: &mut [RoomType; 100]| {
            for i in from.min(99)..(from + n).min(100) {
                results[i as usize] = r;
            }
        };
        let _ = elite_size;
        fill_range(fill, monster_size, RoomType::Monster, &mut results);
        fill += monster_size;
        fill_range(fill, shop_size, RoomType::Shop, &mut results);
        fill += shop_size;
        fill_range(fill, treasure_size, RoomType::Treasure, &mut results);
        let mut choice = results[(roll * 100.0) as usize];
        if force_chest {
            choice = RoomType::Treasure;
        }
        let _ = elite;
        if choice == RoomType::Elite {
            self.event_chances[0] = 0.0;
        } else {
            self.event_chances[0] += 0.1;
        }
        if choice == RoomType::Monster {
            if self.has_relic(RelicId::JuzuBracelet) {
                choice = RoomType::Event;
            }
            self.event_chances[1] = 0.1;
        } else {
            self.event_chances[1] += 0.1;
        }
        if choice == RoomType::Shop {
            self.event_chances[2] = 0.03;
        } else {
            self.event_chances[2] += 0.03;
        }
        if choice == RoomType::Treasure {
            self.event_chances[3] = 0.02;
        } else {
            self.event_chances[3] += 0.02;
        }
        choice
    }

    /// The current room is the burning elite (`currMapNode.hasEmeraldKey`).
    fn on_emerald_elite(&self) -> bool {
        self.config.final_act
            && self.room == Some(RoomType::Elite)
            && self.pos.is_some_and(|(x, y)| (y as usize) < self.map.rows.len() && self.map.node(x, y).has_emerald_key)
    }

    /// Training: starts the fight `key` in a room of type `room` (Monster, Elite or Boss) from
    /// the run's current state, as `start_combat` does on entering such a room. The screen is
    /// Combat afterwards, or Unsupported if the engine can't build the encounter.
    pub fn start_fight(&mut self, key: &str, room: RoomType) {
        self.room = Some(room);
        self.event_elite = false;
        self.start_combat(key);
    }

    /// The modded game's `harness` command (tools/trace/CommunicationMod.patch) starting the
    /// fight `key`: an ordinary monster room on the next floor, entered as `nextRoomTransition`
    /// enters one (the floor's RNG seeds, the relics' room hooks), whatever the encounter. For
    /// watching training fights in the game (python/watch/watch_practice.py).
    pub fn start_harness_fight(&mut self, key: &str) {
        self.floor += 1;
        self.rng.enter_floor(self.floor);
        self.event_monsters = None;
        self.stance = crate::combat::Stance::Neutral;
        self.relics_on_enter_room(RoomType::Monster);
        self.room = Some(RoomType::Monster);
        self.event_elite = false;
        self.relics_just_entered_room(RoomType::Monster);
        self.start_combat(key);
    }

    fn start_combat(&mut self, key: &str) {
        let rng = std::mem::replace(&mut self.rng, Rngs::new(0));
        let monsters = match self.event_monsters.take() {
            Some((k, m)) if k == key => Some(m),
            _ => None,
        };
        match Combat::from_run(
            rng,
            self.config.ascension,
            key,
            &self.deck,
            self.hp,
            self.max_hp,
            self.gold,
            self.relics.clone(),
            self.next_uuid,
            self.next_instance,
            self.room == Some(RoomType::Elite) || self.event_elite,
            if self.on_emerald_elite() { Some(self.act) } else { None },
            self.config.character,
            monsters,
            self.stance,
        ) {
            Ok(mut c) => {
                c.potions = self.potions.clone();
                self.encounter = Some(key.to_string());
                self.combat = Some(Box::new(c));
                self.screen = Screen::Combat;
                if self.combat.as_ref().unwrap().is_over() {
                    self.finish_combat();
                }
            }
            Err(e) => {
                self.screen = Screen::Unsupported { what: e };
            }
        }
    }

    /// `Colosseum.reopen`: back at the event after the first fight, the game resets the player
    /// (`AbstractDungeon.resetPlayer`: the stance back to Neutral, so the Nobs fight doesn't start
    /// in the Slavers fight's last stance; run H14H of the 2026-10-28 Watcher soak) and runs its
    /// `preBattlePrep` again. Nothing that sets up lasts (the Nobs fight prepares everything
    /// afresh), but its RNG calls do: `initializeDeck` shuffles a draw pile (one shuffle seed),
    /// and the relics' `atPreBattle` runs, where Enchiridion rolls a random Power card.
    fn colosseum_reopen_prep(&mut self) {
        self.stance = crate::combat::Stance::Neutral;
        let _ = self.rng.shuffle.random_long();
        if self.has_relic(RelicId::Enchiridion) {
            let n = crate::combat::truly_random_in_combat_list(self.config.character, crate::cards::CardType::Power).len();
            let _ = self.rng.card_random.random(n as i32 - 1);
        }
    }

    /// End of combat: copy state back and build the reward screen (`AbstractRoom.update` COMPLETE).
    fn finish_combat(&mut self) {
        let c = *self.combat.take().unwrap();
        self.rng = c.rng.clone();
        self.hp = c.player.hp;
        self.max_hp = c.player.max_hp;
        self.gold = c.player.gold;
        self.relics = c.player.relics.clone();
        self.next_uuid = c.next_uuid;
        self.next_instance = c.next_instance;
        self.potions = c.potions.clone();
        self.stance = c.player.stance;
        // Lesson Learned's upgrades.
        for &uuid in &c.master_deck_upgrades {
            if let Some(card) = self.deck.iter_mut().find(|d| d.uuid == uuid) {
                card.upgrade();
            }
        }
        for &id in &c.master_deck_add {
            let card = self.new_card(id);
            self.add_obtained_card(card);
        }
        for &(uuid, amount) in &c.master_deck_growth {
            for card in self.deck.iter_mut().filter(|d| d.uuid == uuid) {
                card.misc += amount;
                if card.id == CardId::GeneticAlgorithm {
                    card.base_block = card.misc;
                } else {
                    card.base_damage = card.misc;
                }
            }
        }
        if c.outcome == Some(Outcome::Defeat) {
            self.outcome = Some(RunOutcome::Died);
            self.screen = Screen::Over;
            return;
        }
        // MonsterGroup.haveMonstersEscaped(): every monster ran away.
        let escaped = c.monsters.iter().all(|m| m.is_escaping);
        let mut items = Vec::new();
        // Looter and Mugger die() add what they stole to the rewards.
        let stolen: i32 = c
            .monsters
            .iter()
            .filter(|m| m.is_dying)
            .map(|m| match m.ai {
                crate::monsters::Ai::Looter { stolen_gold, .. } | crate::monsters::Ai::Mugger { stolen_gold, .. } => stolen_gold,
                _ => 0,
            })
            .sum();
        if stolen > 0 {
            items.push(Reward::StolenGold(stolen));
        }
        let room = self.room.unwrap_or(RoomType::Monster);
        if room == RoomType::Event {
            // An event fight: the event's own rewards, then the EventRoom potion chance and a
            // card reward; no gold or relic drop of its own.
            self.event_elite = false;
            if let Some(e) = self.event_resume.take() {
                // rewardAllowed is off: the potion roll still happens, then the event returns.
                let mut discarded = Vec::new();
                self.add_potion_reward(&mut discarded, true);
                self.event_rewards_pending = None;
                self.colosseum_reopen_prep();
                self.screen = Screen::Event { event: e };
                return;
            }
            let mut items = self.event_rewards_pending.take().unwrap_or_default();
            self.add_potion_reward(&mut items, true);
            if c.smoked {
                self.screen = Screen::Rewards { items: Vec::new() };
                return;
            }
            let cards = self.reward_cards(RewardRoom::Normal);
            if !cards.is_empty() {
                items.push(Reward::Cards(cards));
            }
            self.screen = Screen::Rewards { items };
            return;
        }
        match room {
            RoomType::Boss => {
                let mut g = 100 + self.rng.misc.random_range(-5, 5);
                if self.config.ascension >= 13 {
                    g = map::java_round(g as f32 * 0.75);
                }
                Run::add_gold(&mut items, g);
                if self.act_data().id == "TheBeyond" {
                    // No reward screen after the Act 3 boss; the proceed button moves on.
                    self.after_beyond_boss();
                    return;
                }
                if self.act_data().id == "TheEnding" {
                    // The Heart is beaten: TrueVictoryRoom.
                    self.outcome = Some(RunOutcome::Won);
                    self.screen = Screen::Over;
                    return;
                }
            }
            RoomType::Elite => {
                let g = self.rng.treasure.random_range(25, 35);
                Run::add_gold(&mut items, g);
            }
            _ => {
                if !escaped {
                    let g = self.rng.treasure.random_range(10, 20);
                    Run::add_gold(&mut items, g);
                }
            }
        }
        if room == RoomType::Elite {
            let roll = self.rng.relic.random_range(0, 99);
            let tier = if roll < 50 {
                RelicTier::Common
            } else if roll > 82 {
                RelicTier::Rare
            } else {
                RelicTier::Uncommon
            };
            let r = self.random_relic(tier);
            items.push(Reward::Relic(r));
            if self.has_relic(RelicId::BlackStar) {
                let roll = self.rng.relic.random_range(0, 99);
                let tier = if roll < 50 {
                    RelicTier::Common
                } else if roll > 82 {
                    RelicTier::Rare
                } else {
                    RelicTier::Uncommon
                };
                let r = self.random_noncamp_relic(tier);
                items.push(Reward::Relic(r));
            }
            // addEmeraldKey
            if !self.keys.emerald && !items.is_empty() && self.on_emerald_elite() {
                items.push(Reward::EmeraldKey);
            }
        }
        self.add_potion_reward(&mut items, !escaped || room == RoomType::Elite);
        let reward_room = match room {
            RoomType::Elite => RewardRoom::Elite,
            RoomType::Boss => RewardRoom::Boss,
            _ => RewardRoom::Normal,
        };
        if c.smoked {
            // Smoke Bomb: the rolls above still happen, but the reward screen opens empty.
            self.screen = Screen::Rewards { items: Vec::new() };
            return;
        }
        let cards = self.reward_cards(reward_room);
        if !cards.is_empty() {
            items.push(Reward::Cards(cards));
        }
        if room == RoomType::Monster && self.has_relic(RelicId::PrayerWheel) {
            let cards = self.reward_cards(reward_room);
            if !cards.is_empty() {
                items.push(Reward::Cards(cards));
            }
        }
        self.screen = Screen::Rewards { items };
    }

    /// TreasureRoom: `getRandomChest` on entry, opened straight away.
    fn open_chest(&mut self) {
        let roll = self.rng.treasure.random_range(0, 99);
        let (common, uncommon, gold_chance, gold_amt) = if roll < 50 {
            (75, 25, 50, 25)
        } else if roll < 83 {
            (35, 50, 35, 50)
        } else {
            (0, 75, 50, 75)
        };
        let roll = self.rng.treasure.random_range(0, 99);
        let gold = roll < gold_chance;
        let tier = if roll < common {
            RelicTier::Common
        } else if roll < uncommon + common {
            RelicTier::Uncommon
        } else {
            RelicTier::Rare
        };
        let mut items = Vec::new();
        self.relics_on_chest_open(&mut items);
        if gold {
            let g = self.rng.treasure.random_float_between(gold_amt as f32 * 0.9, gold_amt as f32 * 1.1);
            Run::add_gold(&mut items, map::java_round(g));
        }
        let r = self.random_relic(tier);
        items.push(Reward::Relic(r));
        if self.config.final_act && !self.keys.sapphire {
            items.push(Reward::SapphireKey);
        }
        self.relics_on_chest_open_after(&mut items);
        self.screen = Screen::Rewards { items };
    }

    /// TreasureRoomBoss: three boss relics.
    fn enter_boss_chest(&mut self) {
        // nextRoomTransition out of the boss room (a MonsterRoom) trims the monster list too.
        if !self.monster_list.is_empty() {
            self.monster_list.remove(0);
        } else {
            let mut r = self.rng.monster.clone();
            self.generate_strong(&mut r, 12);
            self.rng.monster = r;
        }
        self.floor += 1;
        self.rng.enter_floor(self.floor);
        self.relics_on_enter_room(RoomType::Treasure);
        self.room = Some(RoomType::Treasure);
        let relics = (0..3).map(|_| self.random_relic(RelicTier::Boss)).collect();
        self.screen = Screen::BossRelic { relics };
    }

    /// Lists the encounter names this port supports.
    pub fn supported_encounters() -> Vec<&'static str> {
        encounters::all()
    }
}

fn screen_name(s: &Screen) -> &'static str {
    match s {
        Screen::Neow { .. } => "neow",
        Screen::Map => "map",
        Screen::Combat => "combat",
        Screen::Rewards { .. } => "rewards",
        Screen::CardReward { .. } => "card reward",
        Screen::Grid { .. } => "grid select",
        Screen::Rest => "rest site",
        Screen::BossRelic { .. } => "boss relic",
        Screen::Shop { .. } => "shop",
        Screen::Event { .. } => "event",
        Screen::Unsupported { .. } => "unsupported room",
        Screen::Over => "game over",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// After the last Act 3 boss the game enters the VictoryRoom (the Heart's door) on a floor of
    /// its own, with that floor's RNG seeds and Maw Bank's gold, both on to Act 4 with the keys and
    /// for a win without them (playback of an A10 Heart win, 2026-10-25: the engine's Act 4 floors
    /// were one behind and its hands drawn with another shuffle seed).
    #[test]
    fn the_hearts_door_is_a_floor_of_its_own() {
        for keys in [true, false] {
            let mut run = Run::new(RunConfig::new(7));
            run.act = 3;
            run.floor = 50;
            run.relics.push(RelicState::new(RelicId::MawBank));
            run.keys = Keys { ruby: keys, emerald: keys, sapphire: keys };
            let gold = run.gold;
            run.after_beyond_boss();
            assert_eq!(run.floor, 51);
            assert_eq!(run.gold, gold + 12);
            let mut seeded = sts_rng::StsRandom::new(7 + 51);
            assert_eq!(run.rng.shuffle.random_long(), seeded.random_long());
            if keys {
                assert_eq!((run.act, run.outcome), (4, None));
            } else {
                assert_eq!(run.outcome, Some(RunOutcome::Won));
            }
        }
    }

    /// The game's `Colosseum.reopen` runs `preBattlePrep` between the two fights: one shuffle
    /// seed, and Enchiridion's Power card roll. Seen in four soak runs, where the Nobs fight's
    /// opening hand only matched the game's with the extra shuffle.
    /// Addict's rob, The Mausoleum and Big Fish's box start the curse's obtain effect before the
    /// relic arrives: a relic gained in that click doesn't let Omamori cancel the curse, but does
    /// react to it (soak run GBHL: the Addict's Omamori, and the game kept the Shame).
    #[test]
    fn a_curse_started_before_its_relic_is_not_cancelled_by_it() {
        let shames = |run: &Run| run.deck.iter().filter(|c| c.id == CardId::Shame).count();
        let mut run = Run::new(RunConfig::new(1));
        let shame = run.new_card(CardId::Shame);
        run.obtain_card_then_relic(shame, RelicId::Omamori);
        assert_eq!(shames(&run), 1);
        assert_eq!(run.relics.iter().find(|r| r.id == RelicId::Omamori).unwrap().counter, 2);
        // An Omamori already held still cancels it.
        let shame = run.new_card(CardId::Shame);
        run.obtain_card(shame);
        assert_eq!(shames(&run), 1);

        let mut run = Run::new(RunConfig::new(1));
        let gold = run.gold;
        let shame = run.new_card(CardId::Shame);
        run.obtain_card_then_relic(shame, RelicId::CeramicFish);
        assert_eq!(run.gold, gold + 9);
    }

    /// Events that make their monsters before the fight roll their HP then, as the game's
    /// `monsterHpRng` shows (recorded Defect run, 2026-10-19: the Mushrooms' Stomp rolled the
    /// three Fungi Beasts on that click, the engine had waited for Fight).
    #[test]
    fn event_monsters_are_rolled_when_the_event_makes_them() {
        let mut run = Run::new(RunConfig::new(1));
        let before = run.rng.monster_hp.counter;
        run.open_event(events::EventId::Mushrooms);
        assert_eq!(run.rng.monster_hp.counter, before);
        run.step(&RunAction::EventChoice { index: 0 }).unwrap();
        assert_eq!(run.rng.monster_hp.counter, before + 3);
        let hp: Vec<i32> = run.event_monsters.as_ref().unwrap().1.iter().map(|m| m.hp).collect();
        run.step(&RunAction::EventChoice { index: 0 }).unwrap();
        let c = run.combat.as_ref().unwrap();
        assert_eq!(c.monsters.iter().map(|m| m.hp).collect::<Vec<_>>(), hp);
        assert_eq!(c.rng.monster_hp.counter, before + 3);

        // Masked Bandits make theirs on opening, fight or not.
        let mut run = Run::new(RunConfig::new(1));
        let before = run.rng.monster_hp.counter;
        run.open_event(events::EventId::MaskedBandits);
        assert!(run.rng.monster_hp.counter > before);
    }

    /// Parasite takes 3 max HP when it leaves the deck (`onRemoveFromMasterDeck`); Ascender's
    /// Bane, Curse of the Bell and Necronomicurse can't be removed (the purge lists leave them
    /// out).
    #[test]
    fn parasite_costs_max_hp_on_removal_and_some_curses_stay() {
        let mut run = Run::new(RunConfig::new(1));
        let p = run.new_card(CardId::Parasite);
        run.deck.push(p);
        let max = run.max_hp;
        let i = run.deck.len() - 1;
        run.take_cards(&[i]);
        assert_eq!(run.max_hp, max - 3);
        for id in [CardId::AscendersBane, CardId::CurseOfTheBell, CardId::Necronomicurse, CardId::Parasite] {
            let c = run.new_card(id);
            run.deck.push(c);
        }
        let removable: Vec<CardId> = neow::purgeable(&run).into_iter().map(|i| run.deck[i].id).collect();
        assert!(removable.contains(&CardId::Parasite));
        assert!(!removable.iter().any(|id| matches!(id, CardId::AscendersBane | CardId::CurseOfTheBell | CardId::Necronomicurse)));
    }

    /// Necronomicon brings its curse with it, through the usual obtain (soak run IN7N: Cursed
    /// Tome's book, after which the game drew a Necronomicurse the engine didn't have).
    #[test]
    fn necronomicon_adds_necronomicurse() {
        let curses = |run: &Run| run.deck.iter().filter(|c| c.id == CardId::Necronomicurse).count();
        let mut run = Run::new(RunConfig::new(1));
        run.obtain_relic(RelicId::Necronomicon);
        assert_eq!(curses(&run), 1);
        let mut run = Run::new(RunConfig::new(1));
        run.obtain_relic(RelicId::Omamori);
        run.obtain_relic(RelicId::Necronomicon);
        assert_eq!(curses(&run), 0);
        assert_eq!(run.relics.iter().find(|r| r.id == RelicId::Omamori).unwrap().counter, 1);
    }

    /// A grid closing in the shop with rewards left from a Cauldron bought there brings the
    /// reward screen back (`closeCurrentScreen`, case GRID; soak run KZPH froze the harness).
    #[test]
    fn a_shop_grid_reopens_rewards_left_in_the_shop() {
        let mut run = Run::new(RunConfig::new(1));
        run.gold = 999;
        run.room = Some(RoomType::Shop);
        run.open_shop();
        run.obtain_relic(RelicId::Cauldron);
        let Screen::Rewards { items } = &run.screen else { panic!("Cauldron's rewards") };
        assert_eq!(items.len(), 5);
        run.step(&RunAction::TakeReward { index: 0 }).unwrap();
        run.step(&RunAction::Proceed).unwrap();
        assert!(matches!(run.screen, Screen::Shop { .. }));
        let deck = run.deck.len();
        run.step(&RunAction::Purge).unwrap();
        run.step(&RunAction::Select { indices: vec![0] }).unwrap();
        assert_eq!(run.deck.len(), deck - 1);
        let Screen::Rewards { items } = &run.screen else { panic!("the 4 potions left, again") };
        assert_eq!(items.len(), 4);
        assert!(run.rewards_labelled());
        run.step(&RunAction::Proceed).unwrap();
        assert!(matches!(run.screen, Screen::Shop { .. }));
        // Leaving the shop forgets them.
        run.step(&RunAction::Proceed).unwrap();
        assert!(run.shop_leftover_rewards.is_empty());
    }

    /// The shop's removal leaves out a bottled card (soak run NBG6: Bottled Tornado's Combust).
    #[test]
    fn shop_removal_skips_bottled_cards() {
        let mut run = Run::new(RunConfig::new(1));
        run.gold = 999;
        run.room = Some(RoomType::Shop);
        run.open_shop();
        let bottled = run.deck.len() - 1;
        run.deck[bottled].bottled = true;
        run.step(&RunAction::Purge).unwrap();
        let Screen::Grid { options, .. } = &run.screen else { panic!("the removal grid") };
        assert!(!options.contains(&bottled));
        assert_eq!(options.len(), run.deck.len() - 1);
    }

    /// Ascension 14 takes 5 max HP from the Ironclad, and from A6 a run starts at 90% (soak run
    /// NBGD at A17: 68 HP in the game, 67 in the engine, which took 6).
    #[test]
    fn ascension_starting_hp() {
        let hp = |a| {
            let mut cfg = RunConfig::new(1);
            cfg.ascension = a;
            let r = Run::new(cfg);
            (r.max_hp, r.hp)
        };
        assert_eq!(hp(0), (80, 80));
        assert_eq!(hp(6), (80, 72));
        assert_eq!(hp(14), (75, 68));
        assert_eq!(hp(20), (75, 68));
        // Ascender's Bane is the deck's first card (soak runs NBH7, NBLA: different hands).
        let mut cfg = RunConfig::new(1);
        cfg.ascension = 10;
        let r = Run::new(cfg);
        assert_eq!(r.deck[0].id, CardId::AscendersBane);
        assert_eq!(r.deck.len(), 11);
    }

    /// The shop's removal discounts don't compound: the later one (Membership Card) wins (soak
    /// run GBNN: 38 in the game with both, where the engine had 30).
    #[test]
    fn shop_removal_discounts_do_not_compound() {
        let mut run = Run::new(RunConfig::new(1));
        assert_eq!(run.purge_price(), 75);
        run.obtain_relic(RelicId::Courier);
        assert_eq!(run.purge_price(), 60);
        run.obtain_relic(RelicId::MembershipCard);
        assert_eq!(run.purge_price(), 38);
        run.purge_cost = 100;
        assert_eq!(run.purge_price(), 50);
        run.obtain_relic(RelicId::SmilingMask);
        assert_eq!(run.purge_price(), 50);
    }

    #[test]
    fn colosseum_reopen_uses_a_shuffle_seed_and_enchiridions_roll() {
        let mut run = Run::new(RunConfig::new(1));
        let (shuffle, card_random) = (run.rng.shuffle.counter, run.rng.card_random.counter);
        run.colosseum_reopen_prep();
        assert_eq!((run.rng.shuffle.counter, run.rng.card_random.counter), (shuffle + 1, card_random));
        run.obtain_relic(RelicId::Enchiridion);
        run.colosseum_reopen_prep();
        assert_eq!((run.rng.shuffle.counter, run.rng.card_random.counter), (shuffle + 2, card_random + 1));
    }

    /// The Slavers fight's last stance doesn't carry into the Nobs fight (`resetPlayer`), while a
    /// stance kept from a fight otherwise lasts until the next room.
    #[test]
    fn colosseum_reopen_resets_the_stance() {
        let mut run = Run::new(RunConfig::new(1));
        run.stance = crate::combat::Stance::Wrath;
        run.colosseum_reopen_prep();
        assert_eq!(run.stance, crate::combat::Stance::Neutral);
    }
}
