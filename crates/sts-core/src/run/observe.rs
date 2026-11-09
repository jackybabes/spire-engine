//! What an agent sees of a run: the current screen, the run state and the legal actions.

use serde::{Deserialize, Serialize};

use super::map::RoomType;
use super::{GridPurpose, Reward, Run, RunAction, RunOutcome, Screen};
use crate::cards::Card;
use crate::combat::observe::Observation;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeckCard {
    pub id: String,
    pub name: String,
    pub upgraded: bool,
    pub cost: i32,
    pub card_type: String,
    pub rarity: String,
    /// The card's own numbers: Searing Blow's damage after each upgrade, a grown Ritual Dagger
    /// or Genetic Algorithm.
    #[serde(default)]
    pub base_damage: i32,
    #[serde(default)]
    pub base_block: i32,
    #[serde(default)]
    pub base_magic: i32,
}

impl DeckCard {
    pub fn of(c: &Card) -> DeckCard {
        DeckCard {
            id: c.data().java_id.to_string(),
            name: c.name(),
            upgraded: c.upgraded,
            cost: c.cost,
            card_type: format!("{:?}", c.card_type()),
            rarity: format!("{:?}", c.data().rarity),
            base_damage: c.base_damage,
            base_block: c.base_block,
            base_magic: c.base_magic,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MapNodeView {
    pub x: i32,
    pub y: i32,
    pub room: Option<RoomType>,
    pub edges: Vec<(i32, i32)>,
    /// The burning elite, which carries the Emerald key (with the final act on).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub emerald: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RewardView {
    Gold { amount: i32 },
    StolenGold { amount: i32 },
    Relic { id: String, name: String },
    Potion { id: String, name: String },
    Cards { cards: Vec<DeckCard> },
    EmeraldKey,
    SapphireKey,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ScreenView {
    Neow { options: Vec<String> },
    Map { next: Vec<MapChoice> },
    Combat { combat: Box<Observation> },
    Rewards { items: Vec<RewardView> },
    CardReward { cards: Vec<DeckCard> },
    Grid { purpose: GridPurpose, options: Vec<usize>, count: usize },
    Rest,
    BossRelic { relics: Vec<String> },
    Shop { cards: Vec<Option<ShopItemView>>, relics: Vec<Option<ShopItemView>>, potions: Vec<Option<ShopItemView>>, purge_cost: Option<i32> },
    /// An event: `stage` (which of its screens), `n` (the numbers it rolled: heal amounts, gold,
    /// damage...), and what it asks for: We Meet Again's potion and card, N'loth's relics. The
    /// viewer writes what each option will do from them.
    Event {
        id: String,
        options: Vec<EventOptionView>,
        cards: Vec<Option<DeckCard>>,
        #[serde(default)]
        stage: u8,
        #[serde(default)]
        n: [i32; 4],
        #[serde(default)]
        potion: Option<String>,
        #[serde(default)]
        give_card: Option<String>,
        #[serde(default)]
        relics: Vec<String>,
        /// Match and Keep: each card's place on its 4 x 3 board (column + 4 * row).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        board: Vec<u8>,
    },
    Unsupported { what: String },
    Over { outcome: Option<RunOutcome> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShopItemView {
    pub id: String,
    pub name: String,
    pub price: i32,
    /// For cards.
    pub card: Option<DeckCard>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventOptionView {
    pub label: String,
    pub enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MapChoice {
    pub x: i32,
    pub y: i32,
    pub room: Option<RoomType>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunObservation {
    pub act: i32,
    pub floor: i32,
    pub hp: i32,
    pub max_hp: i32,
    pub gold: i32,
    pub deck: Vec<DeckCard>,
    pub relics: Vec<String>,
    pub potions: Vec<Option<String>>,
    pub boss: String,
    pub keys: super::Keys,
    pub pos: Option<(i32, i32)>,
    pub map: Vec<MapNodeView>,
    pub screen: ScreenView,
    pub legal_actions: Vec<RunAction>,
    /// The run's RNG counters (the fight's own during combat) and card rarity offset, to line up
    /// with the game's (`tools/trace/playback.py` logs them as `game rng`).
    pub rng: std::collections::BTreeMap<String, i32>,
}

fn reward_view(r: &Reward) -> RewardView {
    match r {
        Reward::Gold(g) => RewardView::Gold { amount: *g },
        Reward::StolenGold(g) => RewardView::StolenGold { amount: *g },
        Reward::Relic(id) => RewardView::Relic { id: id.java_id().to_string(), name: id.data().name.to_string() },
        Reward::Potion(p) => RewardView::Potion { id: p.data().java_id.to_string(), name: p.data().name.to_string() },
        Reward::Cards(cs) => RewardView::Cards { cards: cs.iter().map(DeckCard::of).collect() },
        Reward::EmeraldKey => RewardView::EmeraldKey,
        Reward::SapphireKey => RewardView::SapphireKey,
    }
}

impl Run {
    pub fn observe(&self) -> RunObservation {
        let screen = match &self.screen {
            Screen::Neow { options } => ScreenView::Neow { options: options.iter().map(|o| format!("{:?} / {:?}", o.reward, o.drawback)).collect() },
            Screen::Map => {
                let y = self.pos.map_or(0, |(_, y)| y + 1);
                let next = self
                    .next_nodes()
                    .into_iter()
                    .map(|x| MapChoice { x, y, room: if y >= super::map::MAP_HEIGHT as i32 { Some(RoomType::Boss) } else { self.map.node(x, y).room } })
                    .collect();
                ScreenView::Map { next }
            }
            Screen::Combat => match &self.combat {
                Some(c) => ScreenView::Combat { combat: Box::new(c.observe()) },
                None => ScreenView::Unsupported { what: "combat missing".into() },
            },
            Screen::Rewards { items } => ScreenView::Rewards { items: items.iter().map(reward_view).collect() },
            Screen::CardReward { cards, .. } => ScreenView::CardReward { cards: cards.iter().map(DeckCard::of).collect() },
            Screen::Grid { purpose, options, count } => ScreenView::Grid { purpose: *purpose, options: options.clone(), count: *count },
            Screen::Rest => ScreenView::Rest,
            Screen::BossRelic { relics } => ScreenView::BossRelic { relics: relics.iter().map(|r| r.java_id().to_string()).collect() },
            Screen::Shop { shop } => ScreenView::Shop {
                cards: shop
                    .cards
                    .iter()
                    .map(|c| c.as_ref().map(|c| ShopItemView { id: c.card.data().java_id.into(), name: c.card.name(), price: c.price, card: Some(DeckCard::of(&c.card)) }))
                    .collect(),
                relics: shop
                    .relics
                    .iter()
                    .map(|r| r.as_ref().map(|r| ShopItemView { id: r.relic.java_id().into(), name: r.relic.data().name.into(), price: r.price, card: None }))
                    .collect(),
                potions: shop
                    .potions
                    .iter()
                    .map(|p| p.as_ref().map(|p| ShopItemView { id: p.potion.data().java_id.into(), name: p.potion.data().name.into(), price: p.price, card: None }))
                    .collect(),
                purge_cost: shop.purge_available.then_some(shop.purge_cost),
            },
            Screen::Event { event } => ScreenView::Event {
                id: event.id.key().to_string(),
                options: event.labels.iter().zip(&event.options).map(|(l, &e)| EventOptionView { label: l.clone(), enabled: e }).collect(),
                // Match and Keep only shows cards that have been turned over.
                cards: event
                    .cards
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let shown = event.id != super::events::EventId::MatchAndKeep || event.list.get(i) == Some(&1);
                        shown.then(|| DeckCard::of(c))
                    })
                    .collect(),
                stage: event.screen,
                n: event.n,
                potion: event.potion_slot.and_then(|i| self.potions.get(i).copied().flatten()).map(|p| p.data().name.to_string()),
                give_card: event.card_index.and_then(|i| self.deck.get(i)).map(|c| c.name()),
                relics: event.relics.iter().map(|r| r.data().java_id.to_string()).collect(),
                board: event.board.clone(),
            },
            Screen::Unsupported { what } => ScreenView::Unsupported { what: what.clone() },
            Screen::Over => ScreenView::Over { outcome: self.outcome },
        };
        let map = self
            .map
            .rows
            .iter()
            .flatten()
            .filter(|n| !n.edges.is_empty() || !n.parents.is_empty())
            .map(|n| MapNodeView { x: n.x, y: n.y, room: n.room, edges: n.edges.clone(), emerald: n.has_emerald_key })
            .collect();
        RunObservation {
            act: self.act,
            floor: self.floor,
            hp: self.hp,
            max_hp: self.max_hp,
            gold: self.gold,
            deck: self.deck.iter().map(DeckCard::of).collect(),
            relics: self.relics.iter().map(|r| r.id.java_id().to_string()).collect(),
            potions: self.potions.iter().map(|p| p.map(|p| p.data().java_id.to_string())).collect(),
            boss: self.boss.clone(),
            keys: self.keys,
            pos: self.pos,
            map,
            screen,
            legal_actions: self.legal_actions(),
            rng: {
                let r = self.combat.as_ref().map_or(&self.rng, |c| &c.rng);
                let mut m: std::collections::BTreeMap<String, i32> = [
                    ("card", &r.card), ("misc", &r.misc), ("merchant", &r.merchant), ("potion", &r.potion), ("relic", &r.relic),
                    ("event", &r.event), ("treasure", &r.treasure), ("monster", &r.monster),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.counter))
                .collect();
                m.insert("card_blizz".into(), self.card_blizz);
                m
            },
        }
    }
}
