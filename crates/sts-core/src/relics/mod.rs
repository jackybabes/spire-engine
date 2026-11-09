//! Relics: static data (generated) and per-run relic state.

mod data;

pub use data::{RelicId, BLUE_LIBRARY_ORDER, GREEN_LIBRARY_ORDER, PURPLE_LIBRARY_ORDER, RED_LIBRARY_ORDER, RELIC_DATA, SHARED_LIBRARY_ORDER};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelicTier {
    Starter,
    Common,
    Uncommon,
    Rare,
    Shop,
    Boss,
    Special,
    Deprecated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelicClass {
    Shared,
    Red,
    Green,
    Blue,
    Purple,
}

#[derive(Clone, Copy, Debug)]
pub struct RelicData {
    pub id: RelicId,
    pub java_id: &'static str,
    pub name: &'static str,
    pub tier: RelicTier,
    pub class: RelicClass,
}

/// The other characters' relics whose logic is written so far (milestone 5; the port checklist,
/// `tools/extract/port_checklist.py`, reads this list). Shared and Ironclad relics are all ported.
pub const PORTED: &[RelicId] = &[
    // The Silent.
    RelicId::HoveringKite,
    RelicId::NinjaScroll,
    RelicId::PaperCrane,
    RelicId::RingOfTheSerpent,
    RelicId::SnakeRing,
    RelicId::SneckoSkull,
    RelicId::TheSpecimen,
    RelicId::Tingsha,
    RelicId::ToughBandages,
    RelicId::TwistedFunnel,
    RelicId::WristBlade,
    // The Defect.
    RelicId::CrackedCore,
    RelicId::DataDisk,
    RelicId::EmotionChip,
    RelicId::FrozenCore,
    RelicId::GoldPlatedCables,
    RelicId::Inserter,
    RelicId::NuclearBattery,
    RelicId::RunicCapacitor,
    RelicId::SymbioticVirus,
    // The Watcher.
    RelicId::CloakClasp,
    RelicId::Damaru,
    RelicId::GoldenEye,
    RelicId::HolyWater,
    RelicId::Melange,
    RelicId::PureWater,
    RelicId::TeardropLocket,
    RelicId::VioletLotus,
    RelicId::Duality,
];

impl RelicId {
    pub fn data(self) -> &'static RelicData {
        &RELIC_DATA[self as usize]
    }

    pub fn java_id(self) -> &'static str {
        self.data().java_id
    }

    pub fn from_java_id(id: &str) -> Option<RelicId> {
        RELIC_DATA.iter().find(|d| d.java_id == id).map(|d| d.id)
    }

    /// The starter relic this boss relic replaces, taking its slot (BossRelicSelectScreen).
    pub fn replaces_starter(self) -> Option<RelicId> {
        match self {
            RelicId::BlackBlood => Some(RelicId::BurningBlood),
            RelicId::RingOfTheSerpent => Some(RelicId::SnakeRing),
            RelicId::FrozenCore => Some(RelicId::CrackedCore),
            RelicId::HolyWater => Some(RelicId::PureWater),
            _ => None,
        }
    }

    pub fn from_name(name: &str) -> Option<RelicId> {
        let norm = |s: &str| s.to_ascii_lowercase().replace([' ', '_', '\'', '-'], "");
        let n = norm(name);
        RELIC_DATA
            .iter()
            .find(|d| norm(d.java_id) == n || norm(d.name) == n || norm(&format!("{:?}", d.id)) == n)
            .map(|d| d.id)
    }
}

/// A relic the player owns. `counter` mirrors `AbstractRelic.counter` (-1 when unused).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelicState {
    pub id: RelicId,
    pub counter: i32,
    /// `usedUp` / greyed out.
    pub used_up: bool,
    /// Per-relic booleans (`firstTurn`, `activated`, ...); see the `FLAG_*` constants.
    #[serde(default)]
    pub flags: u8,
    /// The master-deck card a Bottled relic holds.
    #[serde(default)]
    pub card_uuid: Option<u32>,
}

/// `firstTurn` (Ancient Tea Set, Art of War, Lantern, Pocketwatch), `activated` (Gambling Chip,
/// Necronomicon), `isActive` (Red Skull), `trigger` (Orichalcum), `canDraw` (Unceasing Top),
/// `usedThisCombat` (Centennial Puzzle), `ATTACK` (Orange Pellets).
pub const FLAG_A: u8 = 1;
/// `gainEnergyNext` (Art of War), `disabledUntilEndOfTurn` (Unceasing Top), `SKILL` (Orange Pellets).
pub const FLAG_B: u8 = 2;
/// `POWER` (Orange Pellets).
pub const FLAG_C: u8 = 4;

impl RelicState {
    /// A freshly constructed relic, with the counter its constructor sets.
    pub fn new(id: RelicId) -> RelicState {
        use RelicId::*;
        let counter = match id {
            Circlet | NlothsMask => 1,
            Girya | InkBottle | Nunchaku | PenNib | Inserter => 0,
            Matryoshka | Omamori => 2,
            NeowsLament | WingBoots => 3,
            _ => -1,
        };
        RelicState { id, counter, used_up: false, flags: 0, card_uuid: None }
    }

    pub fn flag(&self, f: u8) -> bool {
        self.flags & f != 0
    }

    pub fn set_flag(&mut self, f: u8, on: bool) {
        if on {
            self.flags |= f;
        } else {
            self.flags &= !f;
        }
    }
}
