//! Act 1 events and shrines (`com.megacrit.cardcrawl.events.exordium` / `.shrines`), chosen by
//! `AbstractDungeon.generateEvent`. Act 2 events live in `events_city.rs`.

#[path = "events_beyond.rs"]
mod beyond;
#[path = "events_city.rs"]
mod city;

use serde::{Deserialize, Serialize};
use sts_rng::{java_shuffle, JavaRandom, StsRandom};

use super::relics::TransformRng;
use super::{GridPurpose, Reward, Run, RunOutcome, Screen};
use crate::cards::{Card, CardId, CardRarity, CardType};
use crate::potions::PotionId;
use crate::relics::RelicId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventId {
    BigFish,
    Cleric,
    DeadAdventurer,
    GoldenIdol,
    GoldenWing,
    WorldOfGoop,
    LiarsGame,
    LivingWall,
    Mushrooms,
    ScrapOoze,
    ShiningLight,
    MatchAndKeep,
    GoldenShrine,
    Transmogrifier,
    Purifier,
    UpgradeShrine,
    WheelOfChange,
    AccursedBlacksmith,
    Bonfire,
    FountainOfCleansing,
    FaceTrader,
    Lab,
    NoteForYourself,
    WeMeetAgain,
    WomanInBlue,
    // Act 2.
    Addict,
    BackToBasics,
    Beggar,
    Colosseum,
    CursedTome,
    DrugDealer,
    ForgottenAltar,
    Ghosts,
    MaskedBandits,
    Nest,
    TheLibrary,
    TheMausoleum,
    Vampires,
    Designer,
    Duplicator,
    KnowingSkull,
    Nloth,
    TheJoust,
    // Act 3.
    Falling,
    MindBloom,
    MoaiHead,
    MysteriousSphere,
    SecretPortal,
    SensoryStone,
    TombRedMask,
    WindingHalls,
}

impl EventId {
    /// Every event, in declaration order (the order the encodings number them).
    pub fn all() -> impl Iterator<Item = EventId> {
        // SAFETY: a field-less enum with contiguous discriminants 0..=WindingHalls; transmute checks
        // at compile time that it is one byte.
        (0..=EventId::WindingHalls as u8).map(|i| unsafe { std::mem::transmute::<u8, EventId>(i) })
    }
}

impl EventId {
    /// The game's event key (`EventHelper.getEvent`).
    pub fn key(self) -> &'static str {
        use EventId::*;
        match self {
            BigFish => "Big Fish",
            Cleric => "The Cleric",
            DeadAdventurer => "Dead Adventurer",
            GoldenIdol => "Golden Idol",
            GoldenWing => "Golden Wing",
            WorldOfGoop => "World of Goop",
            LiarsGame => "Liars Game",
            LivingWall => "Living Wall",
            Mushrooms => "Mushrooms",
            ScrapOoze => "Scrap Ooze",
            ShiningLight => "Shining Light",
            MatchAndKeep => "Match and Keep!",
            GoldenShrine => "Golden Shrine",
            Transmogrifier => "Transmorgrifier",
            Purifier => "Purifier",
            UpgradeShrine => "Upgrade Shrine",
            WheelOfChange => "Wheel of Change",
            AccursedBlacksmith => "Accursed Blacksmith",
            Bonfire => "Bonfire Elementals",
            FountainOfCleansing => "Fountain of Cleansing",
            FaceTrader => "FaceTrader",
            Lab => "Lab",
            NoteForYourself => "NoteForYourself",
            WeMeetAgain => "WeMeetAgain",
            WomanInBlue => "The Woman in Blue",
            Addict => "Addict",
            BackToBasics => "Back to Basics",
            Beggar => "Beggar",
            Colosseum => "Colosseum",
            CursedTome => "Cursed Tome",
            DrugDealer => "Drug Dealer",
            ForgottenAltar => "Forgotten Altar",
            Ghosts => "Ghosts",
            MaskedBandits => "Masked Bandits",
            Nest => "Nest",
            TheLibrary => "The Library",
            TheMausoleum => "The Mausoleum",
            Vampires => "Vampires",
            Designer => "Designer",
            Duplicator => "Duplicator",
            KnowingSkull => "Knowing Skull",
            Nloth => "N'loth",
            TheJoust => "The Joust",
            Falling => "Falling",
            MindBloom => "MindBloom",
            MoaiHead => "The Moai Head",
            MysteriousSphere => "Mysterious Sphere",
            SecretPortal => "SecretPortal",
            SensoryStone => "SensoryStone",
            TombRedMask => "Tomb of Lord Red Mask",
            WindingHalls => "Winding Halls",
        }
    }

    pub fn from_key(key: &str) -> Option<EventId> {
        use EventId::*;
        [
            BigFish, Cleric, DeadAdventurer, GoldenIdol, GoldenWing, WorldOfGoop, LiarsGame, LivingWall, Mushrooms, ScrapOoze, ShiningLight,
            MatchAndKeep, GoldenShrine, Transmogrifier, Purifier, UpgradeShrine, WheelOfChange, AccursedBlacksmith, Bonfire, FountainOfCleansing,
            FaceTrader, Lab, NoteForYourself, WeMeetAgain, WomanInBlue, Addict, BackToBasics, Beggar, Colosseum, CursedTome, DrugDealer,
            ForgottenAltar, Ghosts, MaskedBandits, Nest, TheLibrary, TheMausoleum, Vampires, Designer, Duplicator, KnowingSkull, Nloth, TheJoust,
            Falling, MindBloom, MoaiHead, MysteriousSphere, SecretPortal, SensoryStone, TombRedMask, WindingHalls,
        ]
        .into_iter()
        .find(|e| e.key() == key)
    }
}

/// An event in progress: which screen it is on, which options are offered (true = enabled),
/// and the numbers its constructor rolled.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub screen: u8,
    pub options: Vec<bool>,
    /// Short English descriptions of `options`.
    pub labels: Vec<String>,
    /// Event-specific numbers (heal amounts, gold, chances, ...).
    pub n: [i32; 4],
    /// Cards the event holds (Match and Keep's grid, Note For Yourself's card).
    pub cards: Vec<Card>,
    /// Dead Adventurer's remaining rewards (0 gold, 1 nothing, 2 relic); Match and Keep's
    /// face-up flags.
    pub list: Vec<u8>,
    /// We Meet Again: potion slot and deck index on offer.
    pub potion_slot: Option<usize>,
    pub card_index: Option<usize>,
    /// Match and Keep: the first card turned this attempt.
    pub picked: Option<usize>,
    /// N'loth: the two relics on offer.
    #[serde(default)]
    pub relics: Vec<RelicId>,
    /// Match and Keep: each card's place on the board (parallel to `cards`). `placeCards` puts
    /// the card at index i in column i % 4 and row i % 3; CommunicationMod numbers the face-down
    /// cards by column + 4 * row.
    #[serde(default)]
    pub board: Vec<u8>,
}

impl Event {
    /// An event on its first screen with nothing set up (the advisor fills in what it knows).
    pub fn new(id: EventId) -> Event {
        Event {
            id,
            screen: 0,
            options: Vec::new(),
            labels: Vec::new(),
            n: [0; 4],
            cards: Vec::new(),
            list: Vec::new(),
            potion_slot: None,
            card_index: None,
            picked: None,
            relics: Vec::new(),
            board: Vec::new(),
        }
    }

    fn set(&mut self, opts: &[(&str, bool)]) {
        self.options = opts.iter().map(|o| o.1).collect();
        self.labels = opts.iter().map(|o| o.0.to_string()).collect();
    }

    fn leave(&mut self, screen: u8) {
        self.screen = screen;
        self.set(&[("leave", true)]);
    }
}

/// `Exordium`'s event list, shrine list and special one-time events.
impl Run {
    fn is_cursed(&self) -> bool {
        self.deck.iter().any(|c| c.card_type() == CardType::Curse && !matches!(c.id, CardId::Necronomicurse | CardId::CurseOfTheBell | CardId::AscendersBane))
    }

    /// Opens event `id` on its first screen, as entering its room does (for a run rebuilt in a
    /// human player's state at an event: `sts-py`'s human decisions).
    pub fn start_event(&mut self, id: EventId) {
        self.room = Some(super::map::RoomType::Event);
        self.open_event(id);
    }

    /// `AbstractDungeon.generateEvent(rng)` with `new Random(seed, eventRng.counter)`: the
    /// key of the event to open.
    pub(super) fn generate_event(&mut self) -> String {
        let mut rng = StsRandom::with_counter(self.config.seed, self.rng.event.counter);
        if rng.random_float_range(1.0) < 0.25 {
            if !self.shrine_list.is_empty() || !self.special_events.is_empty() {
                self.get_shrine(&mut rng)
            } else {
                self.get_event(&mut rng)
            }
        } else {
            self.get_event(&mut rng)
        }
    }

    fn get_shrine(&mut self, rng: &mut StsRandom) -> String {
        let act = self.act_data().id;
        let city = act == "TheCity";
        let city_or_beyond = city || act == "TheBeyond";
        let mut tmp: Vec<String> = self.shrine_list.clone();
        for e in &self.special_events {
            let ok = match e.as_str() {
                "Fountain of Cleansing" => self.is_cursed(),
                "Designer" => city_or_beyond && self.gold >= 75,
                "Duplicator" => city_or_beyond,
                "FaceTrader" => city || act == "Exordium",
                "Knowing Skull" => city && self.hp > 12,
                "N'loth" => city && self.relics.len() >= 2,
                "The Joust" => city && self.gold >= 50,
                "The Woman in Blue" => self.gold >= 50,
                // Needs CardCrawlGame.playtime >= 800 seconds; a run reaching Act 3 has
                // almost always been played that long, so the port treats it as met.
                "SecretPortal" => act == "TheBeyond",
                _ => true,
            };
            if ok {
                tmp.push(e.clone());
            }
        }
        if tmp.is_empty() {
            // The game would crash here (Random.nextInt(0)); no event is left to give.
            return "none".into();
        }
        let key = tmp[rng.random(tmp.len() as i32 - 1) as usize].clone();
        self.shrine_list.retain(|s| *s != key);
        self.special_events.retain(|s| *s != key);
        key
    }

    fn get_event(&mut self, rng: &mut StsRandom) -> String {
        let golden_idol = self.has_relic(RelicId::GoldenIdol);
        let tmp: Vec<String> = self
            .event_list
            .iter()
            .filter(|e| match e.as_str() {
                "Dead Adventurer" | "Mushrooms" => self.floor > 6,
                "The Moai Head" => golden_idol || !(self.hp as f32 / self.max_hp as f32 > 0.5),
                "The Cleric" => self.gold >= 35,
                "Beggar" => self.gold >= 75,
                "Colosseum" => self.pos.is_some_and(|(_, y)| y > super::map::MAP_HEIGHT as i32 / 2),
                _ => true,
            })
            .cloned()
            .collect();
        if tmp.is_empty() {
            return self.get_shrine(rng);
        }
        let key = tmp[rng.random(tmp.len() as i32 - 1) as usize].clone();
        self.event_list.retain(|s| *s != key);
        key
    }

    fn purgeable_unbottled(&self) -> Vec<usize> {
        super::neow::purgeable(self).into_iter().filter(|&i| !self.deck[i].bottled).collect()
    }

    fn upgradable(&self) -> Vec<usize> {
        (0..self.deck.len()).filter(|&i| self.deck[i].can_upgrade()).collect()
    }

    /// `loseGold` outside a shop.
    fn event_lose_gold(&mut self, amount: i32) {
        self.gold = (self.gold - amount).max(0);
    }

    /// `AbstractPlayer.damage` outside combat: Boot, Torii and Tungsten Rod still apply, and
    /// Fairy in a Bottle or Lizard Tail can save the player.
    pub(super) fn event_damage(&mut self, amount: i32, owner: bool, hp_loss: bool) {
        let mut dmg = amount.max(0);
        if owner && !hp_loss && self.has_relic(RelicId::Boot) && dmg > 0 && dmg < 5 {
            dmg = 5;
        }
        if owner && !hp_loss && self.has_relic(RelicId::Torii) && dmg > 1 && dmg <= 5 {
            dmg = 1;
        }
        if self.has_relic(RelicId::TungstenRod) && dmg > 0 {
            dmg -= 1;
        }
        if dmg <= 0 {
            return;
        }
        self.hp = (self.hp - dmg).max(0);
        if self.hp < 1 && !self.has_relic(RelicId::MarkOfTheBloom) {
            if let Some(slot) = self.potions.iter().position(|p| *p == Some(PotionId::FairyPotion)) {
                let pct = PotionId::FairyPotion.data().potency * if self.has_relic(RelicId::SacredBark) { 2 } else { 1 };
                let amt = ((self.max_hp as f32 * (pct as f32 / 100.0)) as i32).max(1);
                self.heal(amt);
                self.potions[slot] = None;
                return;
            }
            if let Some(r) = self.relics.iter_mut().find(|r| r.id == RelicId::LizardTail && r.counter == -1) {
                r.counter = -2;
                r.used_up = true;
                let amt = (self.max_hp / 2).max(1);
                self.heal(amt);
                return;
            }
        }
        if self.hp < 1 {
            self.outcome = Some(RunOutcome::Died);
            self.screen = Screen::Over;
        }
    }

    /// Builds the event (its constructor) and shows it.
    pub(super) fn open_event(&mut self, id: EventId) {
        use EventId::*;
        let mut e = Event::new(id);
        let asc15 = self.config.ascension >= 15;
        match id {
            BigFish => {
                e.n[0] = self.max_hp / 3;
                e.set(&[("banana: heal", true), ("donut: +5 max HP", true), ("box: relic and Regret", true)]);
            }
            Cleric => {
                e.n[1] = if asc15 { 75 } else { 50 };
                e.n[0] = if self.gold >= 35 { (self.max_hp as f32 * 0.25) as i32 } else { 0 };
                e.set(&[("heal for 35 gold", self.gold >= 35), ("purify: remove a card", self.gold >= e.n[1]), ("leave", true)]);
            }
            DeadAdventurer => {
                let mut rewards = vec![0u8, 1, 2];
                let seed = self.rng.misc.random_long();
                java_shuffle(&mut rewards, &mut JavaRandom::new(seed));
                e.list = rewards;
                e.n[0] = if asc15 { 35 } else { 25 };
                e.n[1] = self.rng.misc.random_range(0, 2);
                e.set(&[("search", true), ("leave", true)]);
            }
            GoldenIdol => {
                let (d, m) = if asc15 { (0.35, 0.1) } else { (0.25, 0.08) };
                e.n[0] = (self.max_hp as f32 * d) as i32;
                e.n[1] = ((self.max_hp as f32 * m) as i32).max(1);
                e.set(&[("take the idol", true), ("leave", true)]);
            }
            GoldenWing => {
                let can = self.deck.iter().any(|c| c.card_type() == CardType::Attack && c.base_damage >= 10);
                e.n[0] = can as i32;
                e.set(&[("pray: lose 7 HP, remove a card", true), ("destroy: 50-80 gold", can), ("leave", true)]);
            }
            WorldOfGoop => {
                let mut loss = if asc15 { self.rng.misc.random_range(35, 75) } else { self.rng.misc.random_range(20, 50) };
                loss = loss.min(self.gold);
                e.n[0] = loss;
                e.set(&[("gather gold: 75 gold, lose 11 HP", true), ("leave: lose gold", true)]);
            }
            LiarsGame => {
                e.n[0] = if asc15 { 150 } else { 175 };
                e.set(&[("agree: gold and Doubt", true), ("disagree", true)]);
            }
            LivingWall => {
                let up = !self.upgradable().is_empty();
                e.set(&[("forget: remove a card", true), ("change: transform a card", true), ("grow: upgrade a card", up)]);
            }
            Mushrooms => {
                e.set(&[("stomp: fight", true), ("eat: heal 25%, Parasite", true)]);
            }
            ScrapOoze => {
                e.n[0] = if asc15 { 5 } else { 3 };
                e.n[1] = 25;
                e.set(&[("reach inside", true), ("leave", true)]);
            }
            ShiningLight => {
                e.n[0] = crate::run::map::java_round(self.max_hp as f32 * if asc15 { 0.3 } else { 0.2 });
                let up = !self.upgradable().is_empty();
                e.set(&[("enter: upgrade 2 random cards", up), ("leave", true)]);
            }
            MatchAndKeep => {
                let mut cards = Vec::new();
                for r in [CardRarity::Rare, CardRarity::Uncommon, CardRarity::Common] {
                    let pid = self.pools.random_card(r, &mut self.rng.card);
                    cards.push(self.new_card(pid));
                }
                if asc15 {
                    for _ in 0..2 {
                        let curse = super::neow::random_curse(&mut self.rng.card);
                        cards.push(self.new_card(curse));
                    }
                } else {
                        let id = self.return_colorless_card(CardRarity::Uncommon);
                    cards.push(self.new_card(id));
                    let curse = super::neow::random_curse(&mut self.rng.card);
                    cards.push(self.new_card(curse));
                }
                let start = self.config.character.start_card_for_event();
                cards.push(self.new_card(start));
                let mut copies = Vec::new();
                for c in cards.iter_mut() {
                    self.preview_obtain(c);
                    let mut copy = self.new_card(c.id);
                    for _ in 0..c.times_upgraded {
                        copy.upgrade();
                    }
                    copies.push(copy);
                }
                cards.extend(copies);
                let seed = self.rng.misc.random_long();
                java_shuffle(&mut cards, &mut JavaRandom::new(seed));
                e.list = vec![0; cards.len()];
                e.board = (0..cards.len()).map(|i| (i % 4 + 4 * (i % 3)) as u8).collect();
                e.cards = cards;
                e.n[0] = 5;
                e.set(&[("play", true)]);
            }
            GoldenShrine => {
                e.n[0] = if asc15 { 50 } else { 100 };
                e.set(&[("pray: gold", true), ("desecrate: 275 gold and Regret", true), ("leave", true)]);
            }
            Transmogrifier => e.set(&[("pray: transform a card", true), ("leave", true)]),
            Purifier => e.set(&[("pray: remove a card", true), ("leave", true)]),
            UpgradeShrine => {
                let up = !self.upgradable().is_empty();
                e.set(&[("pray: upgrade a card", up), ("leave", true)]);
            }
            WheelOfChange => {
                e.n[1] = if asc15 { 15 } else { 10 };
                e.set(&[("spin", true)]);
            }
            AccursedBlacksmith => {
                let up = !self.upgradable().is_empty();
                e.set(&[("forge: upgrade a card", up), ("rummage: Warped Tongs and Pain", true), ("leave", true)]);
            }
            Bonfire => e.set(&[("approach", true)]),
            FountainOfCleansing => e.set(&[("drink: remove curses", true), ("leave", true)]),
            FaceTrader => {
                e.n[0] = if asc15 { 50 } else { 75 };
                e.n[1] = (self.max_hp / 10).max(1);
                e.set(&[("approach", true)]);
            }
            Lab => e.set(&[("search", true)]),
            NoteForYourself => {
                // The card left by the previous run (profile preference NOTE_CARD); a fresh
                // profile has Iron Wave.
                let c = self.new_card(CardId::IronWave);
                e.cards = vec![c];
                e.set(&[("continue", true)]);
            }
            WeMeetAgain => {
                let slots: Vec<usize> = (0..self.potions.len()).filter(|&i| self.potions[i].is_some()).collect();
                if !slots.is_empty() {
                    let mut s = slots.clone();
                    let seed = self.rng.misc.random_long();
                    java_shuffle(&mut s, &mut JavaRandom::new(seed));
                    e.potion_slot = Some(s[0]);
                }
                e.n[0] = if self.gold < 50 {
                    0
                } else if self.gold > 150 {
                    self.rng.misc.random_range(50, 150)
                } else {
                    self.rng.misc.random_range(50, self.gold)
                };
                let cands: Vec<usize> =
                    (0..self.deck.len()).filter(|&i| self.deck[i].data().rarity != CardRarity::Basic && self.deck[i].card_type() != CardType::Curse).collect();
                if !cands.is_empty() {
                    let mut s = cands.clone();
                    let seed = self.rng.misc.random_long();
                    java_shuffle(&mut s, &mut JavaRandom::new(seed));
                    e.card_index = Some(s[0]);
                }
                e.set(&[
                    ("give a potion", e.potion_slot.is_some()),
                    ("give gold", e.n[0] != 0),
                    ("give a card", e.card_index.is_some()),
                    ("attack (leave)", true),
                ]);
            }
            WomanInBlue => {
                e.set(&[("20 gold: 1 potion", true), ("30 gold: 2 potions", true), ("40 gold: 3 potions", true), ("leave", true)]);
            }
            _ => self.open_city_event(&mut e),
        }
        self.screen = Screen::Event { event: Box::new(e) };
    }

    /// Re-shows the event after a sub-screen.
    fn show(&mut self, e: Event) {
        self.screen = Screen::Event { event: Box::new(e) };
    }

    /// Opens a grid over the event; the event is resumed when it closes.
    fn event_grid(&mut self, e: Event, purpose: GridPurpose, options: Vec<usize>, count: usize) {
        self.show(e);
        if !options.is_empty() {
            self.push_screen(Screen::Grid { purpose, options, count });
        }
    }

    /// A reward screen that ends the event.
    fn event_rewards(&mut self, items: Vec<Reward>) {
        self.screen_stack.clear();
        self.screen = Screen::Rewards { items };
    }

    fn screenless_relic_reward(&mut self) -> RelicId {
        let tier = self.random_relic_tier();
        self.random_screenless_relic(tier)
    }

    /// Picks option `k` of the event.
    pub(super) fn event_choose(&mut self, mut e: Event, k: usize) -> Result<(), String> {
        if !e.options.get(k).copied().unwrap_or(false) {
            return Err("that option is not available".into());
        }
        use EventId::*;
        let asc15 = self.config.ascension >= 15;
        let id = e.id;
        // Every event's last screen is a single leave option.
        if e.options.len() == 1 && e.labels[0] == "leave" {
            self.screen = Screen::Map;
            return Ok(());
        }
        match (id, e.screen) {
            (BigFish, _) => {
                match k {
                    0 => self.heal(e.n[0]),
                    1 => self.increase_max_hp(5),
                    _ => {
                        let r = self.screenless_relic_reward();
                        let c = self.new_card(CardId::Regret);
                        self.obtain_card_then_relic(c, r);
                    }
                }
                e.leave(1);
            }
            (Cleric, _) => match k {
                0 => {
                    self.event_lose_gold(35);
                    self.heal(e.n[0]);
                    e.leave(1);
                }
                1 => {
                    let opts = self.purgeable_unbottled();
                    e.leave(1);
                    if !opts.is_empty() {
                        self.event_lose_gold(e.n[1]);
                        self.event_grid(e, GridPurpose::Remove, opts, 1);
                        return Ok(());
                    }
                }
                _ => e.leave(1),
            },
            (DeadAdventurer, 0) => {
                if k == 1 {
                    e.leave(3);
                } else if self.rng.misc.random_range(0, 99) < e.n[0] {
                    // FAIL: the elite wakes up.
                    let gold = self.rng.misc.random_range(25, 35);
                    e.n[2] = gold;
                    self.roll_event_monsters(dead_adventurer_encounter(e.n[1]));
                    e.screen = 1;
                    e.set(&[("fight", true)]);
                } else {
                    e.n[3] += 1;
                    e.n[0] += 25;
                    match e.list.remove(0) {
                        0 => self.gain_gold(30),
                        2 => {
                            let r = self.screenless_relic_reward();
                            self.obtain_relic(r);
                        }
                        _ => {}
                    }
                    if e.n[3] == 3 {
                        e.leave(2);
                    } else {
                        e.set(&[("search", true), ("leave", true)]);
                    }
                }
            }
            (DeadAdventurer, _) => {
                let mut items = vec![Reward::Gold(e.n[2])];
                for r in e.list.clone() {
                    match r {
                        0 => Run::add_gold(&mut items, 30),
                        2 => {
                            let tier = self.random_relic_tier();
                            let relic = self.random_relic(tier);
                            items.push(Reward::Relic(relic));
                        }
                        _ => {}
                    }
                }
                self.event_combat(dead_adventurer_encounter(e.n[1]), items, true);
                return Ok(());
            }
            (GoldenIdol, 0) => {
                if k == 0 {
                    let r = if self.has_relic(RelicId::GoldenIdol) { RelicId::Circlet } else { RelicId::GoldenIdol };
                    self.obtain_relic(r);
                    e.screen = 1;
                    e.set(&[("outrun: Injury", true), ("smash: take damage", true), ("hide: lose max HP", true)]);
                } else {
                    e.leave(2);
                }
            }
            (GoldenIdol, _) => {
                match k {
                    0 => {
                        let c = self.new_card(CardId::Injury);
                        self.obtain_card(c);
                    }
                    1 => self.event_damage(e.n[0], false, false),
                    _ => self.decrease_max_hp(e.n[1]),
                }
                e.leave(2);
            }
            (GoldenWing, 0) => match k {
                0 => {
                    self.event_damage(7, true, false);
                    e.screen = 1;
                    e.set(&[("continue", true)]);
                }
                1 => {
                    let g = self.rng.misc.random_range(50, 80);
                    self.gain_gold(g);
                    e.leave(2);
                }
                _ => e.leave(2),
            },
            (GoldenWing, _) => {
                let opts = self.purgeable_unbottled();
                e.leave(2);
                self.event_grid(e, GridPurpose::Remove, opts, 1);
                return Ok(());
            }
            (WorldOfGoop, _) => {
                if k == 0 {
                    self.event_damage(11, true, false);
                    self.gain_gold(75);
                } else {
                    self.event_lose_gold(e.n[0]);
                }
                e.leave(1);
            }
            (LiarsGame, 0) => {
                if k == 0 {
                    e.screen = 1;
                    e.set(&[("continue", true)]);
                } else {
                    e.leave(2);
                }
            }
            (LiarsGame, _) => {
                let c = self.new_card(CardId::Doubt);
                self.obtain_card(c);
                self.gain_gold(e.n[0]);
                e.leave(2);
            }
            (LivingWall, _) => {
                let opts = self.purgeable_unbottled();
                e.leave(1);
                if !opts.is_empty() {
                    match k {
                        0 => self.event_grid(e, GridPurpose::Remove, opts, 1),
                        1 => self.event_grid(e, GridPurpose::EventTransform, opts, 1),
                        _ => {
                            let up = self.upgradable();
                            self.event_grid(e, GridPurpose::Upgrade, up, 1);
                        }
                    }
                    return Ok(());
                }
            }
            (Mushrooms, 0) => {
                if k == 0 {
                    self.roll_event_monsters("The Mushroom Lair");
                    e.screen = 2;
                    e.set(&[("fight", true)]);
                } else {
                    let amt = (self.max_hp as f32 * 0.25) as i32;
                    self.heal(amt);
                    let c = self.new_card(CardId::Parasite);
                    self.obtain_card(c);
                    e.leave(1);
                }
            }
            (Mushrooms, _) => {
                let gold = self.rng.misc.random_range(20, 30);
                let relic = if self.has_relic(RelicId::OddMushroom) { RelicId::Circlet } else { RelicId::OddMushroom };
                self.event_combat("The Mushroom Lair", vec![Reward::Gold(gold), Reward::Relic(relic)], false);
                return Ok(());
            }
            (ScrapOoze, _) => {
                if k == 0 {
                    self.event_damage(e.n[0], false, false);
                    let roll = self.rng.misc.random_range(0, 99);
                    if roll >= 99 - e.n[1] {
                        let r = self.screenless_relic_reward();
                        e.leave(1);
                        self.show(e);
                        self.obtain_relic(r);
                        return Ok(());
                    }
                    e.n[1] += 10;
                    e.n[0] += 1;
                    e.set(&[("reach inside", true), ("leave", true)]);
                } else {
                    e.leave(1);
                }
            }
            (ShiningLight, _) => {
                if k == 0 {
                    self.event_damage(e.n[0], true, false);
                    self.upgrade_random_cards(2);
                }
                e.leave(1);
            }
            (MatchAndKeep, 0) => {
                e.screen = 1;
                e.set(&[("continue", true)]);
            }
            (MatchAndKeep, 1) => {
                e.screen = 2;
                e.options = vec![true; e.cards.len()];
                e.labels = vec!["turn a card".into(); e.cards.len()];
            }
            (MatchAndKeep, _) => {
                e.list[k] = 1;
                match e.picked.take() {
                    None => {
                        e.picked = Some(k);
                        e.options[k] = false;
                    }
                    Some(first) => {
                        if e.cards[first].id == e.cards[k].id {
                            let won = e.cards[first].clone();
                            let (a, b) = if first > k { (first, k) } else { (k, first) };
                            e.cards.remove(a);
                            e.cards.remove(b);
                            e.list.remove(a);
                            e.list.remove(b);
                            if e.board.len() > a {
                                e.board.remove(a);
                                e.board.remove(b);
                            }
                            let c = self.new_card(won.id);
                            let mut c2 = c;
                            for _ in 0..won.times_upgraded {
                                c2.upgrade();
                            }
                            self.obtain_card(c2);
                        }
                        e.n[0] -= 1;
                        if e.n[0] == 0 || e.cards.is_empty() {
                            e.leave(3);
                        } else {
                            e.options = vec![true; e.cards.len()];
                            e.labels = vec!["turn a card".into(); e.cards.len()];
                        }
                    }
                }
            }
            (GoldenShrine, _) => {
                match k {
                    0 => self.gain_gold(e.n[0]),
                    1 => {
                        self.gain_gold(275);
                        let c = self.new_card(CardId::Regret);
                        self.obtain_card(c);
                    }
                    _ => {}
                }
                e.leave(1);
            }
            (Transmogrifier, _) | (Purifier, _) | (UpgradeShrine, _) => {
                e.leave(1);
                if k == 0 {
                    let (purpose, opts) = match id {
                        Transmogrifier => (GridPurpose::EventTransform, self.purgeable_unbottled()),
                        Purifier => (GridPurpose::Remove, self.purgeable_unbottled()),
                        _ => (GridPurpose::Upgrade, self.upgradable()),
                    };
                    self.event_grid(e, purpose, opts, 1);
                    return Ok(());
                }
            }
            (WheelOfChange, 0) => {
                let result = self.rng.misc.random_range(0, 5);
                e.n[0] = result;
                if result == 0 {
                    // setGold: 100, 200 or 300 by act.
                    let gold = match self.act_data().id {
                        "TheCity" => 200,
                        "TheBeyond" => 300,
                        "Exordium" => 100,
                        _ => 0,
                    };
                    self.gain_gold(gold);
                }
                e.screen = 1;
                let label = ["gold", "relic", "full heal", "curse: Decay", "remove a card", "lose HP"][result as usize];
                e.set(&[(label, true)]);
            }
            (WheelOfChange, _) => {
                e.leave(2);
                match e.n[0] {
                    1 => {
                        let r = self.screenless_relic_reward();
                        self.event_rewards(vec![Reward::Relic(r)]);
                        return Ok(());
                    }
                    2 => {
                        let m = self.max_hp;
                        self.heal(m);
                    }
                    3 => {
                        let c = self.new_card(CardId::Decay);
                        self.obtain_card(c);
                    }
                    4 => {
                        let opts = self.purgeable_unbottled();
                        self.event_grid(e, GridPurpose::Remove, opts, 1);
                        return Ok(());
                    }
                    5 => {
                        let amt = (self.max_hp as f32 * (e.n[1] as f32 / 100.0)) as i32;
                        self.event_damage(amt, false, true);
                    }
                    _ => {}
                }
            }
            (AccursedBlacksmith, _) => {
                e.leave(2);
                match k {
                    0 => {
                        let up = self.upgradable();
                        self.event_grid(e, GridPurpose::Upgrade, up, 1);
                        return Ok(());
                    }
                    1 => {
                        let c = self.new_card(CardId::Pain);
                        self.obtain_card(c);
                        self.show(e);
                        self.obtain_relic(RelicId::WarpedTongs);
                        return Ok(());
                    }
                    _ => {}
                }
            }
            (Bonfire, 0) => {
                e.screen = 1;
                e.set(&[("offer a card", true)]);
            }
            (Bonfire, _) => {
                let opts = self.purgeable_unbottled();
                e.leave(2);
                self.event_grid(e, GridPurpose::Bonfire, opts, 1);
                return Ok(());
            }
            (FountainOfCleansing, _) => {
                if k == 0 {
                    for i in (0..self.deck.len()).rev() {
                        let c = &self.deck[i];
                        if c.card_type() == CardType::Curse && !c.bottled && !matches!(c.id, CardId::AscendersBane | CardId::CurseOfTheBell | CardId::Necronomicurse) {
                            self.take_cards(&[i]);
                        }
                    }
                    self.on_master_deck_change();
                }
                e.leave(1);
            }
            (FaceTrader, 0) => {
                e.screen = 1;
                e.set(&[("touch: gold, take damage", true), ("trade: a face", true), ("leave", true)]);
            }
            (FaceTrader, _) => {
                match k {
                    0 => {
                        self.gain_gold(e.n[0]);
                        self.event_damage(e.n[1], false, false);
                    }
                    1 => {
                        let mut ids: Vec<RelicId> = [RelicId::CultistMask, RelicId::FaceOfCleric, RelicId::GremlinMask, RelicId::NlothsMask, RelicId::SsserpentHead]
                            .into_iter()
                            .filter(|r| !self.has_relic(*r))
                            .collect();
                        if ids.is_empty() {
                            ids.push(RelicId::Circlet);
                        }
                        let seed = self.rng.misc.random_long();
                        java_shuffle(&mut ids, &mut JavaRandom::new(seed));
                        e.leave(2);
                        self.show(e);
                        self.obtain_relic(ids[0]);
                        return Ok(());
                    }
                    _ => {}
                }
                e.leave(2);
            }
            (Lab, _) => {
                let n = if asc15 { 2 } else { 3 };
                let items = (0..n).map(|_| Reward::Potion(super::pools::random_potion(self.config.character, &mut self.rng.potion))).collect();
                self.event_rewards(items);
                return Ok(());
            }
            (NoteForYourself, 0) => {
                e.screen = 1;
                e.set(&[("take the card, store one", true), ("leave", true)]);
            }
            (NoteForYourself, _) => {
                e.leave(2);
                if k == 0 {
                    let c = e.cards.remove(0);
                    self.obtain_card(c);
                    let opts = self.purgeable_unbottled();
                    self.event_grid(e, GridPurpose::Remove, opts, 1);
                    return Ok(());
                }
            }
            (WeMeetAgain, _) => {
                match k {
                    0 => {
                        if let Some(s) = e.potion_slot {
                            self.potions[s] = None;
                        }
                    }
                    1 => self.event_lose_gold(e.n[0]),
                    2 => {
                        if let Some(i) = e.card_index {
                            self.take_cards(&[i]);
                            self.on_master_deck_change();
                        }
                    }
                    _ => {}
                }
                e.leave(1);
                if k < 3 {
                    let r = self.screenless_relic_reward();
                    self.show(e);
                    self.obtain_relic(r);
                    return Ok(());
                }
            }
            (WomanInBlue, _) => {
                if k < 3 {
                    self.event_lose_gold([20, 30, 40][k]);
                    let items = (0..=k).map(|_| Reward::Potion(super::pools::random_potion(self.config.character, &mut self.rng.potion))).collect();
                    self.event_rewards(items);
                    return Ok(());
                }
                if asc15 {
                    let amt = (self.max_hp as f32 * 0.05).ceil() as i32;
                    self.event_damage(amt, false, true);
                }
                e.leave(1);
            }
            _ => return self.city_event_choose(e, k),
        }
        if !matches!(self.screen, Screen::Over) {
            self.show(e);
        }
        Ok(())
    }

    /// Shining Light: upgrade two random upgradable cards (miscRng shuffle).
    fn upgrade_random_cards(&mut self, n: usize) {
        let mut cands = self.upgradable();
        let seed = self.rng.misc.random_long();
        java_shuffle(&mut cands, &mut JavaRandom::new(seed));
        for &i in cands.iter().take(n) {
            self.deck[i].upgrade();
        }
    }

    /// A fight started by an event; `rewards` are added before the combat rewards.
    /// `getCurrRoom().monsters = MonsterHelper.getEncounter(key)` ahead of the fight: on opening
    /// Masked Bandits and Mysterious Sphere, on the Mushrooms' Stomp and on Dead Adventurer's
    /// failed search. The monsters' HP is rolled then (`monsterHpRng`), and the fight takes them.
    pub(super) fn roll_event_monsters(&mut self, key: &str) {
        if let Ok(m) = crate::encounters::create(key, &mut self.rng, self.config.ascension) {
            self.event_monsters = Some((key.to_string(), m));
        }
    }

    fn event_combat(&mut self, encounter: &str, rewards: Vec<Reward>, elite: bool) {
        self.event_rewards_pending = Some(rewards);
        self.event_elite = elite;
        self.start_combat(encounter);
    }

    /// Bonfire Spirits: the reward for the offered card's rarity.
    pub(super) fn bonfire_offer(&mut self, card: &Card) {
        match card.data().rarity {
            CardRarity::Curse => {
                let r = if self.has_relic(RelicId::SpiritPoop) { RelicId::Circlet } else { RelicId::SpiritPoop };
                self.obtain_relic(r);
            }
            CardRarity::Common | CardRarity::Special => self.heal(5),
            CardRarity::Uncommon => {
                let m = self.max_hp;
                self.heal(m);
            }
            CardRarity::Rare => {
                self.max_hp += 10;
                let m = self.max_hp;
                self.heal(m);
            }
            _ => {}
        }
    }

    /// The event transform: `transformCard(c, false, miscRng)`.
    pub(super) fn event_transform(&mut self, indices: &[usize]) {
        let removed = self.take_cards(indices);
        for c in removed {
            let id = super::neow::transformed_id(self, c.id, TransformRng::Misc);
            let card = self.new_card(id);
            self.obtain_card(card);
        }
    }

}

/// Dead Adventurer's `getMonster()`.
fn dead_adventurer_encounter(monster: i32) -> &'static str {
    match monster {
        0 => "3 Sentries",
        1 => "Gremlin Nob",
        _ => "Lagavulin Event",
    }
}
