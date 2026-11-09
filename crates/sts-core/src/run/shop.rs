//! The merchant (`Merchant`, `ShopScreen`, `StoreRelic`, `StorePotion`).

use serde::{Deserialize, Serialize};

use super::map::java_round;
use super::pools::{self, RewardRoom};
use super::{GridPurpose, Run, Screen};
use crate::cards::{Card, CardId, CardRarity, CardType};
use crate::potions::{PotionId, PotionRarity};
use crate::relics::{RelicId, RelicTier};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShopCard {
    pub card: Card,
    pub price: i32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShopRelic {
    pub relic: RelicId,
    pub price: i32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShopPotion {
    pub potion: PotionId,
    pub price: i32,
}

/// What is for sale. Sold-out slots are `None`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Shop {
    /// Five colored cards, then two colorless ones.
    pub cards: Vec<Option<ShopCard>>,
    pub relics: Vec<Option<ShopRelic>>,
    pub potions: Vec<Option<ShopPotion>>,
    pub purge_available: bool,
    pub purge_cost: i32,
}

fn card_price(r: CardRarity) -> i32 {
    match r {
        CardRarity::Common => 50,
        CardRarity::Uncommon => 75,
        CardRarity::Rare => 150,
        CardRarity::Basic | CardRarity::Special => 9999,
        _ => 0,
    }
}

pub(super) fn relic_price(t: RelicTier) -> i32 {
    match t {
        RelicTier::Starter => 300,
        RelicTier::Common | RelicTier::Shop => 150,
        RelicTier::Uncommon => 250,
        RelicTier::Rare => 300,
        RelicTier::Special => 400,
        RelicTier::Boss => 999,
        RelicTier::Deprecated => -1,
    }
}

fn potion_price(p: PotionId) -> i32 {
    match p.data().rarity {
        PotionRarity::Common => 50,
        PotionRarity::Uncommon => 75,
        PotionRarity::Rare => 100,
        _ => 999,
    }
}

impl Run {
    /// `AbstractDungeon.getCardFromPool(rarity, type, true)`, rolling the rarity each attempt
    /// the way the merchant does.
    /// A shop card of type `t`: the rarity from cardRng; the card from cardRng too, or for The
    /// Courier's restock (`restock`) from `rng.courier` (unseeded in the game).
    fn shop_card(&mut self, t: CardType, restock: bool) -> CardId {
        let rarity = self.roll_card_rarity(RewardRoom::Shop);
        self.card_from_pool(rarity, t, restock)
    }

    /// `getCardFromPool`, including its switch fall-through and Power fallbacks.
    fn card_from_pool(&mut self, rarity: CardRarity, t: CardType, restock: bool) -> CardId {
        let order = match rarity {
            CardRarity::Rare => vec![CardRarity::Rare, CardRarity::Uncommon, CardRarity::Common],
            CardRarity::Uncommon => vec![CardRarity::Uncommon, CardRarity::Common],
            _ => vec![CardRarity::Common],
        };
        for r in order {
            let mut tmp: Vec<CardId> = self.pools.card_pool(r).iter().copied().filter(|c| c.data().card_type == t).collect();
            if !tmp.is_empty() {
                tmp.sort_by(|a, b| a.data().java_id.cmp(b.data().java_id));
                let rng = if restock { &mut self.rng.courier } else { &mut self.rng.card };
                return tmp[rng.random(tmp.len() as i32 - 1) as usize];
            }
            if t == CardType::Power && r != CardRarity::Rare {
                let up = if r == CardRarity::Common { CardRarity::Uncommon } else { CardRarity::Rare };
                return self.card_from_pool(up, t, restock);
            }
        }
        // Unreached: every rarity has cards of every type a shop asks for.
        self.config.character.starting_deck()[0]
    }

    fn shop_card_entry(&mut self, id: CardId, colorless: bool) -> ShopCard {
        let mut card = self.new_card(id);
        let mut price = card_price(card.data().rarity) as f32 * self.rng.merchant.random_float_between(0.9, 1.1);
        if colorless {
            price *= 1.2;
        }
        self.preview_obtain(&mut card);
        ShopCard { card, price: price as i32 }
    }

    /// `ShopScreen.rollRelicTier()`.
    fn shop_relic_tier(&mut self) -> RelicTier {
        let roll = self.rng.merchant.random(99);
        if roll < 48 {
            RelicTier::Common
        } else if roll < 82 {
            RelicTier::Uncommon
        } else {
            RelicTier::Rare
        }
    }

    /// `returnRandomRelicEnd(tier)`.
    fn random_relic_end(&mut self, tier: RelicTier) -> RelicId {
        let relics = self.relics.clone();
        let deck = self.deck.clone();
        let ctx = pools::SpawnCtx { floor: self.floor, act: self.act, in_shop: true, relics: &relics, deck: &deck };
        self.pools.take_relic_end(tier, &ctx)
    }

    fn discount(&self, price: i32) -> i32 {
        let mut p = price;
        if self.has_relic(RelicId::Courier) {
            p = java_round(p as f32 * 0.8);
        }
        if self.has_relic(RelicId::MembershipCard) {
            p = java_round(p as f32 * 0.5);
        }
        p
    }

    /// `ShopScreen.init`'s card removal price. Unlike the item prices, the discounts don't
    /// compound: `applyDiscount` sets `actualPurgeCost = round(purgeCost * multiplier)` from the
    /// base each time, so with The Courier and Membership Card the later 0.5 wins (75 -> 38, not
    /// 30; soak run GBNN, where the engine counted removal affordable at 31 gold and the game
    /// didn't). Ascension 16's markup leaves removal alone.
    pub(crate) fn purge_price(&self) -> i32 {
        if self.has_relic(RelicId::SmilingMask) {
            50
        } else if self.has_relic(RelicId::MembershipCard) {
            java_round(self.purge_cost as f32 * 0.5)
        } else if self.has_relic(RelicId::Courier) {
            java_round(self.purge_cost as f32 * 0.8)
        } else {
            self.purge_cost
        }
    }

    /// `new Merchant()` and `ShopScreen.init`.
    pub(super) fn open_shop(&mut self) {
        let mut ids: Vec<CardId> = Vec::new();
        for (t, distinct) in [(CardType::Attack, false), (CardType::Attack, true), (CardType::Skill, false), (CardType::Skill, true), (CardType::Power, false)] {
            let mut c = self.shop_card(t, false);
            while distinct && Some(&c) == ids.last() {
                c = self.shop_card(t, false);
            }
            ids.push(c);
        }
        let colorless = [
            self.pools.random_colorless(CardRarity::Uncommon, &mut self.rng.card).unwrap_or(CardId::SwiftStrike),
            self.pools.random_colorless(CardRarity::Rare, &mut self.rng.card).unwrap_or(CardId::Apotheosis),
        ];
        let mut cards: Vec<Option<ShopCard>> = Vec::new();
        for id in ids {
            let e = self.shop_card_entry(id, false);
            cards.push(Some(e));
        }
        for id in colorless {
            let e = self.shop_card_entry(id, true);
            cards.push(Some(e));
        }
        let sale = self.rng.merchant.random_range(0, 4) as usize;
        if let Some(c) = cards[sale].as_mut() {
            c.price /= 2;
        }
        let mut relics = Vec::new();
        for i in 0..3 {
            let tier = if i != 2 { self.shop_relic_tier() } else { RelicTier::Shop };
            let r = self.random_relic_end(tier);
            let price = java_round(relic_price(r.data().tier) as f32 * self.rng.merchant.random_float_between(0.95, 1.05));
            relics.push(Some(ShopRelic { relic: r, price }));
        }
        let mut potions = Vec::new();
        for _ in 0..3 {
            let p = pools::return_random_potion(self.config.character, &mut self.rng.potion);
            let price = java_round(potion_price(p) as f32 * self.rng.merchant.random_float_between(0.95, 1.05));
            potions.push(Some(ShopPotion { potion: p, price }));
        }
        let mut shop = Shop { cards, relics, potions, purge_available: true, purge_cost: self.purge_cost };
        let apply = |shop: &mut Shop, mult: f32| {
            for r in shop.relics.iter_mut().flatten() {
                r.price = java_round(r.price as f32 * mult);
            }
            for p in shop.potions.iter_mut().flatten() {
                p.price = java_round(p.price as f32 * mult);
            }
            for c in shop.cards.iter_mut().flatten() {
                c.price = java_round(c.price as f32 * mult);
            }
        };
        if self.config.ascension >= 16 {
            apply(&mut shop, 1.1);
        }
        if self.has_relic(RelicId::Courier) {
            apply(&mut shop, 0.8);
        }
        if self.has_relic(RelicId::MembershipCard) {
            apply(&mut shop, 0.5);
        }
        shop.purge_cost = self.purge_price();
        self.screen = Screen::Shop { shop: Box::new(shop) };
    }

    /// `AbstractPlayer.loseGold` in a shop: `onSpendGold` stops Maw Bank.
    pub(super) fn lose_gold(&mut self, amount: i32) {
        self.gold = (self.gold - amount).max(0);
        if let Some(r) = self.relics.iter_mut().find(|r| r.id == RelicId::MawBank) {
            if !r.used_up {
                r.counter = -2;
                r.used_up = true;
            }
        }
    }

    pub(super) fn buy_card(&mut self, mut shop: Box<Shop>, i: usize) -> Result<(), String> {
        let entry = shop.cards.get(i).cloned().flatten().ok_or("nothing for sale there")?;
        if self.gold < entry.price {
            return Err("not enough gold".into());
        }
        self.obtain_card(entry.card.clone());
        self.lose_gold(entry.price);
        if !self.has_relic(RelicId::Courier) {
            shop.cards[i] = None;
        } else if i >= 5 {
            let rarity = if self.rng.merchant.random_float() < 0.3 { CardRarity::Rare } else { CardRarity::Uncommon };
            let id = self.pools.random_colorless(rarity, &mut self.rng.card).unwrap_or(CardId::SwiftStrike);
            let mut c = self.new_card(id);
            self.preview_obtain(&mut c);
            let price = self.set_price(&c, true);
            shop.cards[i] = Some(ShopCard { card: c, price });
        } else {
            // The restock uses getCardFromPool(..., useRng = false): unseeded in the game, from
            // rng.courier here (and in the modded game, HarnessCourierPatch).
            let t = entry.card.card_type();
            let id = self.shop_card(t, true);
            let mut c = self.new_card(id);
            self.preview_obtain(&mut c);
            let price = self.set_price(&c, false);
            shop.cards[i] = Some(ShopCard { card: c, price });
        }
        self.screen = Screen::Shop { shop };
        Ok(())
    }

    /// `ShopScreen.setPrice` for a restocked card.
    fn set_price(&mut self, c: &Card, colorless: bool) -> i32 {
        let mut price = card_price(c.data().rarity) as f32 * self.rng.merchant.random_float_between(0.9, 1.1);
        if colorless {
            price *= 1.2;
        }
        if self.has_relic(RelicId::Courier) {
            price *= 0.8;
        }
        if self.has_relic(RelicId::MembershipCard) {
            price *= 0.5;
        }
        price as i32
    }

    pub(super) fn buy_relic(&mut self, mut shop: Box<Shop>, i: usize) -> Result<(), String> {
        let entry = shop.relics.get(i).cloned().flatten().ok_or("nothing for sale there")?;
        if self.gold < entry.price {
            return Err("not enough gold".into());
        }
        self.lose_gold(entry.price);
        if entry.relic == RelicId::MembershipCard {
            for r in shop.relics.iter_mut().flatten() {
                r.price = java_round(r.price as f32 * 0.5);
            }
            for p in shop.potions.iter_mut().flatten() {
                p.price = java_round(p.price as f32 * 0.5);
            }
            for c in shop.cards.iter_mut().flatten() {
                c.price = java_round(c.price as f32 * 0.5);
            }
        }
        // The relic may open its own screen; the shop is resumed afterwards.
        self.screen = Screen::Shop { shop: shop.clone() };
        self.obtain_relic(entry.relic);
        for c in shop.cards.iter_mut().flatten() {
            let mut card = c.card.clone();
            self.preview_obtain(&mut card);
            c.card = card;
        }
        if entry.relic != RelicId::Courier && !self.has_relic(RelicId::Courier) {
            shop.relics[i] = None;
        } else {
            let mut r;
            loop {
                let tier = self.shop_relic_tier();
                r = self.random_relic_end(tier);
                if !matches!(r, RelicId::OldCoin | RelicId::SmilingMask | RelicId::MawBank | RelicId::Courier) {
                    break;
                }
            }
            let price = java_round(relic_price(r.data().tier) as f32 * self.rng.merchant.random_float_between(0.95, 1.05));
            shop.relics[i] = Some(ShopRelic { relic: r, price: self.discount(price) });
        }
        // StoreRelic.purchaseRelic: Membership Card applies its discount to removal again (from the
        // base price), Smiling Mask sets it to 50; The Courier leaves it as it was.
        if matches!(entry.relic, RelicId::MembershipCard | RelicId::SmilingMask) {
            shop.purge_cost = self.purge_price();
        }
        self.put_shop_back(shop);
        Ok(())
    }

    /// Stores the updated shop wherever it now is (the current screen, or under a relic's screen).
    fn put_shop_back(&mut self, shop: Box<Shop>) {
        if let Screen::Shop { shop: s } = &mut self.screen {
            *s = shop;
            return;
        }
        if let Some(Screen::Shop { shop: s }) = self.screen_stack.iter_mut().rev().find(|s| matches!(s, Screen::Shop { .. })) {
            *s = shop;
        }
    }

    pub(super) fn buy_potion(&mut self, mut shop: Box<Shop>, i: usize) -> Result<(), String> {
        let entry = shop.potions.get(i).cloned().flatten().ok_or("nothing for sale there")?;
        if self.has_relic(RelicId::Sozu) {
            return Err("Sozu: no potions".into());
        }
        if self.gold < entry.price {
            return Err("not enough gold".into());
        }
        if !self.obtain_potion(entry.potion) {
            return Err("potion slots are full".into());
        }
        self.lose_gold(entry.price);
        if self.has_relic(RelicId::Courier) {
            let p = pools::return_random_potion(self.config.character, &mut self.rng.potion);
            let price = java_round(potion_price(p) as f32 * self.rng.merchant.random_float_between(0.95, 1.05));
            shop.potions[i] = Some(ShopPotion { potion: p, price: self.discount(price) });
        } else {
            shop.potions[i] = None;
        }
        self.screen = Screen::Shop { shop };
        Ok(())
    }

    /// The cards the shop's removal offers: `getGroupWithoutBottledCards(getPurgeableCards())`, so
    /// not a card in a Bottled Flame, Lightning or Tornado (soak run NBG6 removed its bottled
    /// Combust in the engine; the game's grid didn't have it).
    pub(super) fn shop_purgeable(&self) -> Vec<usize> {
        super::neow::purgeable(self).into_iter().filter(|&i| !self.deck[i].bottled).collect()
    }

    pub(super) fn start_purge(&mut self, shop: Box<Shop>) -> Result<(), String> {
        if !shop.purge_available || self.gold < shop.purge_cost {
            return Err("cannot remove a card now".into());
        }
        let options = self.shop_purgeable();
        if options.is_empty() {
            return Err("no card can be removed".into());
        }
        self.screen = Screen::Shop { shop };
        self.push_screen(Screen::Grid { purpose: GridPurpose::ShopPurge, options, count: 1 });
        Ok(())
    }

    /// `ShopScreen.purgeCard` once the card is chosen.
    pub(super) fn finish_purge(&mut self, indices: &[usize]) {
        let cost = match self.screen_stack.last() {
            Some(Screen::Shop { shop }) => shop.purge_cost,
            _ => self.purge_price(),
        };
        self.lose_gold(cost);
        self.purge_cost += 25;
        self.take_cards(indices);
        self.on_master_deck_change();
        let new_cost = self.purge_price();
        if let Some(Screen::Shop { shop }) = self.screen_stack.last_mut() {
            shop.purge_available = false;
            shop.purge_cost = new_cost;
        }
    }
}
