//! Per-act tables from `Exordium`, `TheCity` and `TheBeyond`: encounter weights, bosses,
//! events, shrines, room chances and card upgrade odds.

use super::map::{RoomChances, EXORDIUM_CHANCES};

pub struct ActData {
    /// `AbstractDungeon.id`.
    pub id: &'static str,
    pub weak: &'static [(&'static str, f32)],
    pub weak_count: usize,
    pub strong: &'static [(&'static str, f32)],
    pub elites: &'static [(&'static str, f32)],
    /// `generateExclusions`: strong encounters the first one may not be, given the last weak one.
    pub exclusions: fn(&str) -> &'static [&'static str],
    /// The boss on a profile that has not seen this act's bosses.
    pub first_boss: &'static str,
    /// The list shuffled with `monsterRng.randomLong()` once all bosses are seen.
    pub bosses: [&'static str; 3],
    pub events: &'static [&'static str],
    pub shrines: &'static [&'static str],
    pub chances: RoomChances,
    /// `mapRng = new Random(seed + actNum * mult)`.
    pub map_seed_mult: i64,
}

impl ActData {
    /// `cardUpgradedChance` from `initializeLevelSpecificChances`.
    pub fn card_upgraded_chance(&self, ascension: i32) -> f32 {
        match (self.id, ascension >= 12) {
            ("TheCity", false) => 0.25,
            ("TheCity", true) => 0.125,
            ("TheBeyond" | "TheEnding", false) => 0.5,
            ("TheBeyond" | "TheEnding", true) => 0.25,
            _ => 0.0,
        }
    }
}

pub const EXORDIUM: ActData = ActData {
    id: "Exordium",
    weak: &[("Cultist", 2.0), ("Jaw Worm", 2.0), ("2 Louse", 2.0), ("Small Slimes", 2.0)],
    weak_count: 3,
    strong: &[
        ("Blue Slaver", 2.0),
        ("Gremlin Gang", 1.0),
        ("Looter", 2.0),
        ("Large Slime", 2.0),
        ("Lots of Slimes", 1.0),
        ("Exordium Thugs", 1.5),
        ("Exordium Wildlife", 1.5),
        ("Red Slaver", 1.0),
        ("3 Louse", 2.0),
        ("2 Fungi Beasts", 2.0),
    ],
    elites: &[("Gremlin Nob", 1.0), ("Lagavulin", 1.0), ("3 Sentries", 1.0)],
    exclusions: |last| match last {
        "Looter" => &["Exordium Thugs"],
        "Blue Slaver" => &["Red Slaver", "Exordium Thugs"],
        "2 Louse" => &["3 Louse"],
        "Small Slimes" => &["Large Slime", "Lots of Slimes"],
        _ => &[],
    },
    first_boss: "The Guardian",
    bosses: ["The Guardian", "Hexaghost", "Slime Boss"],
    events: &["Big Fish", "The Cleric", "Dead Adventurer", "Golden Idol", "Golden Wing", "World of Goop", "Liars Game", "Living Wall", "Mushrooms", "Scrap Ooze", "Shining Light"],
    shrines: &["Match and Keep!", "Golden Shrine", "Transmorgrifier", "Purifier", "Upgrade Shrine", "Wheel of Change"],
    chances: EXORDIUM_CHANCES,
    map_seed_mult: 1,
};

pub const THE_CITY: ActData = ActData {
    id: "TheCity",
    weak: &[("Spheric Guardian", 2.0), ("Chosen", 2.0), ("Shell Parasite", 2.0), ("3 Byrds", 2.0), ("2 Thieves", 2.0)],
    weak_count: 2,
    strong: &[
        ("Chosen and Byrds", 2.0),
        ("Sentry and Sphere", 2.0),
        ("Snake Plant", 6.0),
        ("Snecko", 4.0),
        ("Centurion and Healer", 6.0),
        ("Cultist and Chosen", 3.0),
        ("3 Cultists", 3.0),
        ("Shelled Parasite and Fungi", 3.0),
    ],
    elites: &[("Gremlin Leader", 1.0), ("Slavers", 1.0), ("Book of Stabbing", 1.0)],
    exclusions: |last| match last {
        "Spheric Guardian" => &["Sentry and Sphere"],
        "3 Byrds" => &["Chosen and Byrds"],
        "Chosen" => &["Chosen and Byrds", "Cultist and Chosen"],
        _ => &[],
    },
    first_boss: "Champ",
    bosses: ["Automaton", "Collector", "Champ"],
    events: &[
        "Addict",
        "Back to Basics",
        "Beggar",
        "Colosseum",
        "Cursed Tome",
        "Drug Dealer",
        "Forgotten Altar",
        "Ghosts",
        "Masked Bandits",
        "Nest",
        "The Library",
        "The Mausoleum",
        "Vampires",
    ],
    shrines: &["Match and Keep!", "Wheel of Change", "Golden Shrine", "Transmorgrifier", "Purifier", "Upgrade Shrine"],
    chances: EXORDIUM_CHANCES,
    map_seed_mult: 100,
};

pub const THE_BEYOND: ActData = ActData {
    id: "TheBeyond",
    weak: &[("3 Darklings", 2.0), ("Orb Walker", 2.0), ("3 Shapes", 2.0)],
    weak_count: 2,
    strong: &[
        ("Spire Growth", 1.0),
        ("Transient", 1.0),
        ("4 Shapes", 1.0),
        ("Maw", 1.0),
        ("Sphere and 2 Shapes", 1.0),
        ("Jaw Worm Horde", 1.0),
        ("3 Darklings", 1.0),
        ("Writhing Mass", 1.0),
    ],
    elites: &[("Giant Head", 2.0), ("Nemesis", 2.0), ("Reptomancer", 2.0)],
    exclusions: |last| match last {
        "3 Darklings" => &["3 Darklings"],
        "Orb Walker" => &["Orb Walker"],
        "3 Shapes" => &["4 Shapes"],
        _ => &[],
    },
    first_boss: "Awakened One",
    bosses: ["Awakened One", "Time Eater", "Donu and Deca"],
    events: &["Falling", "MindBloom", "The Moai Head", "Mysterious Sphere", "SensoryStone", "Tomb of Lord Red Mask", "Winding Halls"],
    shrines: &["Match and Keep!", "Wheel of Change", "Golden Shrine", "Transmorgrifier", "Purifier", "Upgrade Shrine"],
    chances: EXORDIUM_CHANCES,
    map_seed_mult: 200,
};

/// Act 4: fixed encounter lists, no events, and a hand-built map (see `Map::the_ending`).
pub const THE_ENDING: ActData = ActData {
    id: "TheEnding",
    weak: &[],
    weak_count: 0,
    strong: &[],
    elites: &[],
    exclusions: |_| &[],
    first_boss: "The Heart",
    bosses: ["The Heart", "The Heart", "The Heart"],
    events: &[],
    shrines: &[],
    chances: EXORDIUM_CHANCES,
    map_seed_mult: 300,
};

/// The tables for each act.
pub fn act_data(act: i32) -> &'static ActData {
    match act {
        2 => &THE_CITY,
        3 => &THE_BEYOND,
        4 => &THE_ENDING,
        _ => &EXORDIUM,
    }
}
