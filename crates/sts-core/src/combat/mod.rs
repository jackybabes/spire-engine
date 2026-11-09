//! A single combat: `GameActionManager` + the combat parts of `AbstractRoom`, `AbstractPlayer`
//! and `MonsterGroup`, driven headlessly. The engine runs until it needs a decision.

mod actions;
mod cards_use;
pub(crate) use cards_use::truly_random_in_combat_list;
mod damage;
pub use damage::Hits;
mod engine;
mod hooks;
pub mod observe;
pub(crate) mod orbs;
pub use orbs::{Orb, OrbId};
mod stance;
pub use stance::Stance;
mod potions;
mod relics;

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::action::{Action, Cr};
use crate::cards::{Card, CardId};
pub use crate::character::Character;
use crate::monsters::Monster;
use crate::powers::{Power, PowerId};
use crate::relics::{RelicId, RelicState};
use crate::rngs::Rngs;

pub use engine::java_floor;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Player {
    pub hp: i32,
    pub max_hp: i32,
    pub block: i32,
    pub powers: Vec<Power>,
    pub relics: Vec<RelicState>,
    pub gold: i32,
    pub energy_master: i32,
    pub master_hand_size: i32,
    pub game_hand_size: i32,
    pub is_dead: bool,
    pub is_bloodied: bool,
    pub damaged_this_combat: i32,
    pub cards_played_this_turn: i32,
    /// Orb slots, front first (`None` is an empty slot); their number is `maxOrbs`.
    #[serde(default)]
    pub orbs: Vec<Option<Orb>>,
    /// The Watcher's stance.
    #[serde(default)]
    pub stance: Stance,
}

impl Player {
    pub fn power(&self, id: PowerId) -> Option<&Power> {
        self.powers.iter().find(|p| p.id == id)
    }

    pub fn has_power(&self, id: PowerId) -> bool {
        self.power(id).is_some()
    }

    pub fn has_relic(&self, r: RelicId) -> bool {
        self.relics.iter().any(|x| x.id == r)
    }
}

/// One entry of `GameActionManager.cardQueue`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueueItem {
    pub card: QueuedCard,
    pub target: Option<usize>,
    pub energy_on_use: i32,
    pub ignore_energy_total: bool,
    pub autoplay: bool,
    pub random_target: bool,
    pub end_turn_autoplay: bool,
}

/// Where a queued card lives. Java holds a reference; the card may be in hand, in limbo, or
/// nowhere at all (Havoc's card after `UnlimboAction`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum QueuedCard {
    EndTurn,
    InHand(u32),
    InLimbo(u32),
    Owned(Box<Card>),
}

impl QueuedCard {
    pub fn instance(&self) -> Option<u32> {
        match self {
            QueuedCard::EndTurn => None,
            QueuedCard::InHand(i) | QueuedCard::InLimbo(i) => Some(*i),
            QueuedCard::Owned(c) => Some(c.instance),
        }
    }
}

/// A selection screen waiting for the agent (hand select / grid select).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingChoice {
    pub kind: ChoiceKind,
    /// Which pile the indices refer to.
    pub pile: ChoicePile,
    /// Number of cards to pick (exact unless `any_number`).
    pub count: usize,
    pub any_number: bool,
    pub can_pick_zero: bool,
    /// Indices into `pile` that may be chosen.
    pub options: Vec<usize>,
    /// The cards on offer when `pile` is `Offer` (discovery-style choices).
    #[serde(default)]
    pub offer: Vec<Card>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChoicePile {
    Hand,
    Discard,
    Exhaust,
    /// Cards that are not in any pile (`PendingChoice::offer`).
    Offer,
    Draw,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChoiceKind {
    Exhaust,
    Discard { end_turn: bool },
    PutOnDeck,
    /// Cards that could not be upgraded were pulled out of the hand while the screen is open.
    Armaments { cannot_upgrade: Vec<Card> },
    DualWield { amount: i32, cannot_duplicate: Vec<Card> },
    DiscardToTopOfDeck,
    /// Exhume cards were pulled out of the exhaust pile while the screen is open.
    Exhume { upgrade: bool, exhumes: Vec<Card> },
    /// Gambling Chip: the chosen hand cards are discarded and redrawn.
    GamblingChip,
    /// Toolbox: the chosen card goes to the hand.
    Discover,
    /// Attack/Skill/Power/Colorless Potion: `amount` copies of the pick, free this turn.
    Discovery { amount: i32 },
    /// Nilry's Codex: the pick (if any) is shuffled into the draw pile.
    Codex,
    /// Liquid Memories (`set_cost`, free this turn) and Hologram: the picks return to the hand.
    DiscardToHand { set_cost: bool },
    /// Secret Technique / Secret Weapon: the pick moves from the draw pile to the hand.
    FromDeckToHand,
    /// Forethought: the picks go to the bottom of the draw pile, free once.
    Forethought,
    /// Nightmare: `amount` copies of the pick come next turn (Night Terror).
    Nightmare { amount: i32 },
    /// Setup: the pick goes on top of the draw pile, free once.
    Setup,
    /// Well-Laid Plans: the picks stay in hand at the end of the turn.
    Retain,
    /// Recycle: the pick is exhausted for energy.
    Recycle,
    /// Seek: the picks go from the draw pile to the hand (`options` in the game's grid order).
    DrawPileToHand,
    /// Scry: any of the top cards (`options`, top first) go to the discard pile.
    Scry,
    /// Wish, Stance Potion: one option card's effect.
    ChooseOne,
    /// Foreign Influence: the pick (if any) goes to the hand, free this turn when upgraded.
    ForeignInfluence { upgraded: bool },
    /// Meditate: the picks return from the discard pile to the hand, retained.
    Meditate,
    /// Omniscience: the pick is played `amount` times (`options` in the game's grid order).
    Omniscience { amount: i32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Victory,
    Defeat,
}

/// A decision the agent can take.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PlayerAction {
    PlayCard { hand_index: usize, target: Option<usize> },
    EndTurn,
    Choose { indices: Vec<usize> },
    UsePotion { slot: usize, target: Option<usize> },
    DiscardPotion { slot: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Combat {
    pub player: Player,
    /// Monster storage; indices are stable ids used by actions and targets.
    pub monsters: Vec<Monster>,
    /// `MonsterGroup` order (ids into `monsters`), which drives iteration and targeting order.
    pub order: Vec<usize>,
    pub hand: Vec<Card>,
    /// Top of the draw pile is the last element (`CardGroup.addToTop` appends).
    pub draw_pile: Vec<Card>,
    pub discard_pile: Vec<Card>,
    pub exhaust_pile: Vec<Card>,
    pub limbo: Vec<Card>,
    pub energy: i32,
    pub actions: VecDeque<Action>,
    pub pre_turn_actions: VecDeque<Action>,
    pub card_queue: VecDeque<QueueItem>,
    pub monster_queue: VecDeque<usize>,
    pub monster_attacks_queued: bool,
    pub turn_has_ended: bool,
    pub end_turn_queued: bool,
    pub turn: i32,
    pub cannot_lose: bool,
    pub skip_monster_turn: bool,
    pub ascension: i32,
    pub rng: Rngs,
    pub pending: Option<PendingChoice>,
    pub outcome: Option<Outcome>,
    pub cards_played_this_combat: i32,
    pub next_uuid: u32,
    pub next_instance: u32,
    /// Gold stolen this combat (Looter and Mugger hits); returned as a reward if they die.
    pub stolen_gold: i32,
    /// `lastDamageTaken` of the player's last `damage()` call (Vampire damage heals by it).
    #[serde(default)]
    pub last_damage_taken: i32,
    /// Ritual Dagger kills: (card uuid, damage gained) to apply to the master deck.
    #[serde(default)]
    pub master_deck_growth: Vec<(u32, i32)>,
    /// Cards added to the master deck during the fight (`AddCardToDeckAction`).
    #[serde(default)]
    pub master_deck_add: Vec<crate::cards::CardId>,
    /// The burning elite's act number (`applyEmeraldEliteBuff`), if this is that fight.
    #[serde(default)]
    pub emerald_act: Option<i32>,
    /// `AbstractPlayer.flipHorizontal`: the player faces left (Spire Shield and Spear).
    #[serde(default)]
    pub player_flipped: bool,
    pub mugged: bool,
    /// `AbstractRoom.eliteTrigger` (elite rooms).
    #[serde(default)]
    pub elite: bool,
    /// Potion slots (`None` is an empty slot).
    #[serde(default)]
    pub potions: Vec<Option<crate::potions::PotionId>>,
    /// Smoke Bomb was used: the fight ends with no rewards.
    #[serde(default)]
    pub smoked: bool,
    /// `TheBombPower.bombIdOffset`; also numbers the Night Terror powers.
    #[serde(default)]
    pub bomb_counter: u32,
    #[serde(default)]
    pub character: Character,
    /// `GameActionManager.totalDiscardedThisTurn`.
    #[serde(default)]
    pub total_discarded_this_turn: i32,
    /// Attacks in `GameActionManager.cardsPlayedThisTurn` (Finisher).
    #[serde(default)]
    pub attacks_played_this_turn: i32,
    /// `DrawCardAction.drawnCards`: the cards the current draw chain drew (Escape Plan, Scrape).
    #[serde(default)]
    pub drawn_cards: Vec<(u32, crate::cards::CardType)>,
    /// `orbsChanneledThisCombat` (Blizzard, Thunder Strike) and `orbsChanneledThisTurn`.
    #[serde(default)]
    pub orbs_channeled_this_combat: Vec<OrbId>,
    #[serde(default)]
    pub orbs_channeled_this_turn: Vec<OrbId>,
    /// Power cards in `cardsPlayedThisCombat` (a new Force Field costs that much less).
    #[serde(default)]
    pub powers_played_this_combat: i32,
    /// The card each Night Terror power (by `uid`) makes next turn (`NightmarePower.card`).
    #[serde(default)]
    pub night_terror_cards: Vec<(u32, Card)>,
    /// `GameActionManager.mantraGained`: Mantra made this combat (Brilliance).
    #[serde(default)]
    pub mantra_gained: i32,
    /// The types of `cardsPlayedThisCombat`, in order (Follow-Up, Crush Joints, Sash Whip,
    /// Sanctity look at the card before).
    #[serde(default)]
    pub played_types: Vec<crate::cards::CardType>,
    /// The master deck as the fight began (Lesson Learned upgrades one of its cards).
    #[serde(default)]
    pub master_deck: Vec<Card>,
    /// Master-deck cards (by uuid) Lesson Learned upgraded, for the run to apply.
    #[serde(default)]
    pub master_deck_upgrades: Vec<u32>,
}

/// How to start a combat.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CombatConfig {
    pub seed: i64,
    pub ascension: i32,
    pub encounter: String,
    /// (card, upgrade count) in master-deck order.
    pub deck: Vec<(CardId, i32)>,
    pub player_hp: i32,
    pub player_max_hp: i32,
    pub relics: Vec<RelicId>,
    /// Floor number used to reseed the per-floor RNG streams.
    pub floor: i32,
    pub gold: i32,
    /// Relic counters, parallel to `relics` (None keeps each relic's starting counter).
    #[serde(default)]
    pub relic_counters: Option<Vec<i32>>,
    #[serde(default)]
    pub potions: Vec<Option<crate::potions::PotionId>>,
    #[serde(default)]
    pub character: Character,
}

impl CombatConfig {
    /// A character's starting deck, HP and relic.
    pub fn new(character: Character, seed: i64, encounter: &str) -> CombatConfig {
        let hp = character.max_hp();
        CombatConfig {
            seed,
            ascension: 0,
            encounter: encounter.to_string(),
            deck: character.starting_deck().into_iter().map(|c| (c, 0)).collect(),
            player_hp: hp,
            player_max_hp: hp,
            relics: vec![character.starting_relic()],
            floor: 1,
            gold: 99,
            relic_counters: None,
            potions: vec![None; 3],
            character,
        }
    }

    /// Ironclad's starting deck and relic.
    pub fn ironclad(seed: i64, encounter: &str) -> CombatConfig {
        CombatConfig::new(Character::Ironclad, seed, encounter)
    }

    /// The Silent's starting deck and relic.
    pub fn silent(seed: i64, encounter: &str) -> CombatConfig {
        CombatConfig::new(Character::Silent, seed, encounter)
    }

    /// The Watcher's starting deck and relic.
    pub fn watcher(seed: i64, encounter: &str) -> CombatConfig {
        CombatConfig::new(Character::Watcher, seed, encounter)
    }
}

impl Combat {
    pub fn new(config: &CombatConfig) -> Result<Combat, String> {
        let mut rng = Rngs::new(config.seed);
        rng.enter_floor(config.floor);
        let mut c = Combat::blank(rng, config.ascension, config.player_hp, config.player_max_hp, config.gold);
        c.character = config.character;
        c.player.relics = config.relics.iter().map(|&r| RelicState::new(r)).collect();
        if let Some(counters) = &config.relic_counters {
            for (r, &n) in c.player.relics.iter_mut().zip(counters) {
                r.counter = n;
            }
        }
        if !config.potions.is_empty() {
            c.potions = config.potions.clone();
        }
        let mut master_deck = Vec::new();
        for &(id, upgrades) in &config.deck {
            let mut card = c.new_card(id);
            for _ in 0..upgrades {
                card.upgrade();
            }
            master_deck.push(card);
        }
        let monsters = crate::encounters::create(&config.encounter, &mut c.rng, c.ascension)?;
        c.order = (0..monsters.len()).collect();
        c.monsters = monsters;
        c.elite = matches!(config.encounter.as_str(), "Gremlin Nob" | "Lagavulin" | "3 Sentries");
        c.apply_relic_stats();
        c.start(master_deck);
        Ok(c)
    }

    /// Energy and hand size from relics' `onEquip` (`energyMaster`, `masterHandSize`).
    fn apply_relic_stats(&mut self) {
        use RelicId::*;
        for r in &self.player.relics {
            match r.id {
                BustedCrown | CoffeeDripper | CursedKey | Ectoplasm | FusionHammer | MarkOfPain | PhilosopherStone | RunicDome | Sozu | VelvetChoker => {
                    self.player.energy_master += 1
                }
                SneckoEye => self.player.master_hand_size += 2,
                RingOfTheSerpent => self.player.master_hand_size += 1,
                _ => {}
            }
        }
        self.player.game_hand_size = self.player.master_hand_size;
    }

    /// Starts a combat inside a run. The run's RNG streams move into the combat and are
    /// handed back afterwards through the public fields. `monsters` are ones an event already
    /// made for this encounter; otherwise they are made here.
    #[allow(clippy::too_many_arguments)]
    pub fn from_run(
        rng: Rngs,
        ascension: i32,
        encounter: &str,
        master_deck: &[Card],
        hp: i32,
        max_hp: i32,
        gold: i32,
        relics: Vec<RelicState>,
        next_uuid: u32,
        next_instance: u32,
        elite: bool,
        emerald_act: Option<i32>,
        character: Character,
        monsters: Option<Vec<Monster>>,
        stance: Stance,
    ) -> Result<Combat, String> {
        let mut c = Combat::blank(rng, ascension, hp, max_hp, gold);
        c.player.stance = stance;
        c.character = character;
        c.player.relics = relics;
        c.next_uuid = next_uuid;
        c.next_instance = next_instance;
        let monsters = match monsters {
            Some(m) => m,
            None => crate::encounters::create(encounter, &mut c.rng, ascension)?,
        };
        c.order = (0..monsters.len()).collect();
        c.monsters = monsters;
        c.elite = elite;
        c.emerald_act = emerald_act;
        c.apply_relic_stats();
        c.start(master_deck.to_vec());
        Ok(c)
    }

    fn blank(rng: Rngs, ascension: i32, hp: i32, max_hp: i32, gold: i32) -> Combat {
        Combat {
            player: Player {
                hp,
                max_hp,
                block: 0,
                powers: Vec::new(),
                relics: Vec::new(),
                gold,
                energy_master: 3,
                master_hand_size: 5,
                game_hand_size: 5,
                is_dead: false,
                is_bloodied: false,
                damaged_this_combat: 0,
                cards_played_this_turn: 0,
                orbs: Vec::new(),
                stance: Stance::Neutral,
            },
            monsters: Vec::new(),
            order: Vec::new(),
            hand: Vec::new(),
            draw_pile: Vec::new(),
            discard_pile: Vec::new(),
            exhaust_pile: Vec::new(),
            limbo: Vec::new(),
            energy: 0,
            actions: VecDeque::new(),
            pre_turn_actions: VecDeque::new(),
            card_queue: VecDeque::new(),
            monster_queue: VecDeque::new(),
            monster_attacks_queued: true,
            turn_has_ended: false,
            end_turn_queued: false,
            turn: 1,
            cannot_lose: false,
            skip_monster_turn: false,
            ascension,
            rng,
            pending: None,
            outcome: None,
            cards_played_this_combat: 0,
            next_uuid: 0,
            next_instance: 0,
            stolen_gold: 0,
            last_damage_taken: 0,
            master_deck_growth: Vec::new(),
            master_deck_add: Vec::new(),
            emerald_act: None,
            player_flipped: false,
            mugged: false,
            elite: false,
            potions: vec![None; 3],
            smoked: false,
            bomb_counter: 0,
            character: Character::Ironclad,
            total_discarded_this_turn: 0,
            attacks_played_this_turn: 0,
            drawn_cards: Vec::new(),
            night_terror_cards: Vec::new(),
            orbs_channeled_this_combat: Vec::new(),
            orbs_channeled_this_turn: Vec::new(),
            powers_played_this_combat: 0,
            mantra_gained: 0,
            played_types: Vec::new(),
            master_deck: Vec::new(),
            master_deck_upgrades: Vec::new(),
        }
    }

    pub fn new_card(&mut self, id: CardId) -> Card {
        let uuid = self.next_uuid;
        self.next_uuid += 1;
        let inst = self.next_instance;
        self.next_instance += 1;
        let mut c = Card::new(id, uuid, inst);
        // Force Field's constructor: 1 less for each Power played this combat.
        if id == CardId::ForceField {
            for _ in 0..self.powers_played_this_combat {
                c.update_cost(-1);
            }
        }
        // Shiv's constructor adds the player's Accuracy.
        if id == CardId::Shiv {
            if let Some(p) = self.player.power(PowerId::Accuracy) {
                c.base_damage = 4 + p.amount;
                c.damage = c.base_damage;
            }
        }
        c
    }

    /// `makeStatEquivalentCopy`: fresh uuid.
    pub fn stat_copy(&mut self, card: &Card) -> Card {
        let mut copy = self.new_card(card.id);
        for _ in 0..card.times_upgraded {
            copy.upgrade();
        }
        copy.upgraded = card.upgraded;
        copy.times_upgraded = card.times_upgraded;
        copy.base_damage = card.base_damage;
        copy.base_block = card.base_block;
        copy.base_magic = card.base_magic;
        copy.cost = card.cost;
        copy.cost_for_turn = card.cost_for_turn;
        copy.misc = card.misc;
        copy.free_to_play_once = card.free_to_play_once;
        copy.bottled = card.bottled;
        if card.id == CardId::Expunger {
            copy.magic = card.magic;
        }
        copy
    }

    /// `makeSameInstanceOf`: same uuid, new object.
    pub fn same_instance_copy(&mut self, card: &Card) -> Card {
        let mut copy = self.stat_copy(card);
        copy.uuid = card.uuid;
        copy
    }

    /// The copy that Double Tap, Duplication, Burst, Echo Form and Amplify play straight from limbo
    /// (`makeSameInstanceOf`): it keeps the original's base block and damage, and nothing counts
    /// again before it plays, so Stack and Mind Blast play with the original's count (taken now, in
    /// `onUseCard`, before the original reaches its pile).
    pub(crate) fn play_copy(&mut self, card: &Card) -> Card {
        let mut copy = self.same_instance_copy(card);
        copy.applied_base = card.applied_base;
        if copy.applied_base.is_none() {
            self.apply_powers_kept(&mut copy);
        }
        copy
    }

    pub fn is_over(&self) -> bool {
        self.outcome.is_some()
    }

    pub fn powers(&self, c: Cr) -> &Vec<Power> {
        match c {
            Cr::Player => &self.player.powers,
            Cr::Monster(i) => &self.monsters[i].powers,
        }
    }

    pub fn powers_mut(&mut self, c: Cr) -> &mut Vec<Power> {
        match c {
            Cr::Player => &mut self.player.powers,
            Cr::Monster(i) => &mut self.monsters[i].powers,
        }
    }

    pub fn has_power(&self, c: Cr, id: PowerId) -> bool {
        self.powers(c).iter().any(|p| p.id == id)
    }

    pub fn power_amount(&self, c: Cr, id: PowerId) -> i32 {
        self.powers(c).iter().find(|p| p.id == id).map_or(0, |p| p.amount)
    }

    pub fn hp(&self, c: Cr) -> i32 {
        match c {
            Cr::Player => self.player.hp,
            Cr::Monster(i) => self.monsters[i].hp,
        }
    }

    pub fn block(&self, c: Cr) -> i32 {
        match c {
            Cr::Player => self.player.block,
            Cr::Monster(i) => self.monsters[i].block,
        }
    }

    /// `isDeadOrEscaped` (the player counts as dead once `isDead` is set).
    pub fn is_dead_or_escaped(&self, c: Cr) -> bool {
        match c {
            Cr::Player => self.player.is_dead,
            Cr::Monster(i) => self.monsters[i].is_dead_or_escaped(),
        }
    }

    pub fn is_dying(&self, c: Cr) -> bool {
        match c {
            Cr::Player => self.player.is_dead,
            Cr::Monster(i) => self.monsters[i].is_dying,
        }
    }

    /// `MonsterGroup.areMonstersBasicallyDead`.
    pub fn monsters_basically_dead(&self) -> bool {
        self.order.iter().all(|&m| self.monsters[m].is_dying || self.monsters[m].is_escaping)
    }

    pub fn push_bot(&mut self, a: Action) {
        self.actions.push_back(a);
    }

    pub fn push_top(&mut self, a: Action) {
        self.actions.push_front(a);
    }
}
