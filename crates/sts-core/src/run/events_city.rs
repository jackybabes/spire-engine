//! Act 2 events (`com.megacrit.cardcrawl.events.city`) and the special one-time events that
//! only appear from Act 2 on (Designer, Duplicator, Knowing Skull, N'loth, The Joust).

use sts_rng::{java_shuffle, JavaRandom};

use super::{Event, EventId};
use crate::cards::{CardId, CardRarity};
use crate::relics::{RelicId, RelicTier};
use crate::run::{GridPurpose, Reward, Run, Screen};

impl Run {
    /// `returnColorlessCard(rarity)`: shuffles the colorless pool with shuffleRng (the order
    /// change persists) and takes its first card of that rarity.
    pub(in crate::run) fn return_colorless_card(&mut self, rarity: CardRarity) -> CardId {
        let seed = self.rng.shuffle.random_long();
        java_shuffle(&mut self.pools.colorless, &mut JavaRandom::new(seed));
        self.pools.colorless.iter().copied().find(|c| c.data().rarity == rarity).unwrap_or(CardId::SwiftStrike)
    }

    /// `AbstractPlayer.loseRelic` with the `onUnequip` effects that touch the deck.
    pub(in crate::run) fn lose_relic(&mut self, id: RelicId) {
        let Some(pos) = self.relics.iter().position(|r| r.id == id) else { return };
        let r = self.relics.remove(pos);
        match id {
            RelicId::Necronomicon => {
                if let Some(i) = self.deck.iter().position(|c| c.id == CardId::Necronomicurse) {
                    self.deck.remove(i);
                }
            }
            RelicId::BottledFlame | RelicId::BottledLightning | RelicId::BottledTornado => {
                if let Some(c) = self.deck.iter_mut().find(|c| Some(c.uuid) == r.card_uuid) {
                    c.bottled = false;
                }
            }
            _ => {}
        }
    }

    fn obtain_new(&mut self, id: CardId) {
        let c = self.new_card(id);
        self.obtain_card(c);
    }

    fn non_bottled(&self) -> usize {
        self.deck.iter().filter(|c| !c.bottled).count()
    }

    /// Shuffles the upgradable cards with miscRng and upgrades the first `n`.
    fn upgrade_random_shuffled(&mut self, n: usize) {
        let mut cands = self.upgradable();
        let seed = self.rng.misc.random_long();
        java_shuffle(&mut cands, &mut JavaRandom::new(seed));
        for &i in cands.iter().take(n) {
            self.deck[i].upgrade();
        }
    }

    /// The constructors of the Act 2 events.
    pub(super) fn open_city_event(&mut self, e: &mut Event) {
        use EventId::*;
        let asc15 = self.config.ascension >= 15;
        match e.id {
            Addict => e.set(&[("pay 85 gold: relic", self.gold >= 85), ("rob: relic and Shame", true), ("leave", true)]),
            BackToBasics => e.set(&[("elegance: remove a card", true), ("simplicity: upgrade Strikes and Defends", true)]),
            Beggar => e.set(&[("give 75 gold: remove a card", self.gold >= 75), ("leave", true)]),
            Colosseum => e.set(&[("continue", true)]),
            CursedTome => {
                e.n[0] = if asc15 { 15 } else { 10 };
                e.set(&[("read", true), ("leave", true)]);
            }
            DrugDealer => {
                let two = super::super::neow::purgeable(self).len() >= 2;
                e.set(&[("obtain J.A.X.", true), ("transform 2 cards", two), ("Mutagenic Strength", true)]);
            }
            ForgottenAltar => {
                let idol = self.has_relic(RelicId::GoldenIdol);
                e.n[0] = crate::run::map::java_round(self.max_hp as f32 * if asc15 { 0.35 } else { 0.25 });
                e.set(&[("offer Golden Idol: Bloody Idol", idol), ("sacrifice: +5 max HP, take damage", true), ("desecrate: Decay", true)]);
            }
            Ghosts => {
                let mut loss = (self.max_hp as f32 * 0.5).ceil() as i32;
                if loss >= self.max_hp {
                    loss = self.max_hp - 1;
                }
                e.n[0] = loss;
                e.set(&[("accept: lose max HP, Apparitions", true), ("refuse", true)]);
            }
            MaskedBandits => {
                self.roll_event_monsters("Masked Bandits");
                e.set(&[("pay all gold", true), ("fight", true)]);
            }
            Nest => {
                e.n[0] = if asc15 { 50 } else { 99 };
                e.set(&[("continue", true)]);
            }
            TheLibrary => {
                e.n[0] = crate::run::map::java_round(self.max_hp as f32 * if asc15 { 0.2 } else { 0.33 });
                e.set(&[("read: pick 1 of 20 cards", true), ("sleep: heal", true)]);
            }
            TheMausoleum => e.set(&[("open the coffin: relic, maybe Writhe", true), ("leave", true)]),
            Vampires => {
                let mut loss = (self.max_hp as f32 * 0.3).ceil() as i32;
                if loss >= self.max_hp {
                    loss = self.max_hp - 1;
                }
                e.n[0] = loss;
                if self.has_relic(RelicId::BloodVial) {
                    e.n[1] = 1;
                    e.set(&[("accept: lose max HP, Strikes become Bites", true), ("give Blood Vial", true), ("refuse", true)]);
                } else {
                    e.set(&[("accept: lose max HP, Strikes become Bites", true), ("refuse", true)]);
                }
            }
            Designer => {
                // n: upgrades one?, clean-up removes?, then the costs and HP loss in n[2..].
                e.n[0] = self.rng.misc.random_boolean() as i32;
                e.n[1] = self.rng.misc.random_boolean() as i32;
                e.list = if asc15 { vec![50, 75, 110, 5] } else { vec![40, 60, 90, 3] };
                e.set(&[("continue", true)]);
            }
            Duplicator => e.set(&[("pray: duplicate a card", true), ("leave", true)]),
            KnowingSkull => {
                e.n = [6, 6, 6, 6];
                e.set(&[("continue", true)]);
            }
            Nloth => {
                let mut ids: Vec<RelicId> = self.relics.iter().map(|r| r.id).collect();
                let seed = self.rng.misc.random_long();
                java_shuffle(&mut ids, &mut JavaRandom::new(seed));
                e.relics = ids.into_iter().take(2).collect();
                let a = format!("trade {}", e.relics[0].data().name);
                let b = format!("trade {}", e.relics.get(1).map_or("?", |r| r.data().name));
                e.set(&[(&a, true), (&b, true), ("leave", true)]);
            }
            TheJoust => e.set(&[("continue", true)]),
            _ => self.open_beyond_event(e),
        }
    }

    /// Picks option `k` of an Act 2 event.
    pub(super) fn city_event_choose(&mut self, mut e: Event, k: usize) -> Result<(), String> {
        use EventId::*;
        let asc15 = self.config.ascension >= 15;
        match (e.id, e.screen) {
            (Addict, _) => {
                match k {
                    0 => {
                        let tier = self.random_relic_tier();
                        let r = self.random_screenless_relic(tier);
                        self.event_lose_gold(85);
                        self.obtain_relic(r);
                    }
                    1 => {
                        let tier = self.random_relic_tier();
                        let r = self.random_screenless_relic(tier);
                        let shame = self.new_card(CardId::Shame);
                        self.obtain_card_then_relic(shame, r);
                    }
                    _ => {
                        self.screen = Screen::Map;
                        return Ok(());
                    }
                }
                e.leave(1);
            }
            (BackToBasics, _) => {
                e.leave(1);
                if k == 0 {
                    let opts = self.purgeable_unbottled();
                    self.event_grid(e, GridPurpose::Remove, opts, 1);
                    return Ok(());
                }
                for c in self.deck.iter_mut() {
                    let d = c.data();
                    if (d.starter_strike || d.starter_defend) && c.can_upgrade() {
                        c.upgrade();
                    }
                }
            }
            (Beggar, 0) => {
                if k == 0 {
                    self.event_lose_gold(75);
                    e.screen = 1;
                    e.set(&[("continue: remove a card", true)]);
                } else {
                    e.leave(2);
                }
            }
            (Beggar, _) => {
                // The removal grid closes straight onto the map.
                let opts = self.purgeable_unbottled();
                self.screen = Screen::Map;
                if !opts.is_empty() {
                    self.push_screen(Screen::Grid { purpose: GridPurpose::Remove, options: opts, count: 1 });
                }
                return Ok(());
            }
            (Colosseum, 0) => {
                e.screen = 1;
                e.set(&[("fight", true)]);
            }
            (Colosseum, 1) => {
                e.screen = 2;
                e.set(&[("cowardice: leave", true), ("victory: fight the Nobs", true)]);
                self.event_resume = Some(Box::new(e));
                self.event_combat("Colosseum Slavers", Vec::new(), false);
                return Ok(());
            }
            (Colosseum, _) => {
                if k == 1 {
                    let rare = self.random_relic(RelicTier::Rare);
                    let uncommon = self.random_relic(RelicTier::Uncommon);
                    let items = vec![Reward::Relic(rare), Reward::Relic(uncommon), Reward::Gold(100)];
                    self.event_combat("Colosseum Nobs", items, true);
                } else {
                    self.screen = Screen::Map;
                }
                return Ok(());
            }
            (CursedTome, 0) => {
                if k == 0 {
                    e.screen = 1;
                    e.set(&[("continue (lose 1 HP)", true)]);
                } else {
                    e.leave(5);
                }
            }
            (CursedTome, s @ 1..=3) => {
                self.event_damage(s as i32, false, true);
                e.screen = s + 1;
                if s < 3 {
                    e.set(&[("continue", true)]);
                } else {
                    let take = format!("take the book (lose {} HP)", e.n[0]);
                    e.set(&[(&take, true), ("stop reading (lose 3 HP)", true)]);
                }
            }
            (CursedTome, _) => {
                if k == 0 {
                    self.event_damage(e.n[0], false, true);
                    if matches!(self.screen, Screen::Over) {
                        return Ok(());
                    }
                    let mut books: Vec<RelicId> = [RelicId::Necronomicon, RelicId::Enchiridion, RelicId::NilrysCodex].into_iter().filter(|r| !self.has_relic(*r)).collect();
                    if books.is_empty() {
                        books.push(RelicId::Circlet);
                    }
                    let r = books[self.rng.misc.random(books.len() as i32 - 1) as usize];
                    self.event_rewards(vec![Reward::Relic(r)]);
                    return Ok(());
                }
                self.event_damage(3, false, true);
                e.leave(5);
            }
            (DrugDealer, _) => {
                e.leave(1);
                match k {
                    0 => self.obtain_new(CardId::JAX),
                    1 => {
                        let opts = super::super::neow::purgeable(self);
                        self.event_grid(e, GridPurpose::EventTransform, opts, 2);
                        return Ok(());
                    }
                    _ => {
                        let r = if self.has_relic(RelicId::MutagenicStrength) { RelicId::Circlet } else { RelicId::MutagenicStrength };
                        self.obtain_relic(r);
                    }
                }
            }
            (ForgottenAltar, _) => {
                match k {
                    0 => {
                        if self.has_relic(RelicId::BloodyIdol) {
                            self.obtain_relic(RelicId::Circlet);
                        } else {
                            // instantObtain(player, idolSlot, false): replaces the Golden Idol.
                            let slot = self.relics.iter().position(|r| r.id == RelicId::GoldenIdol).unwrap_or(0);
                            self.relics[slot] = crate::relics::RelicState::new(RelicId::BloodyIdol);
                        }
                    }
                    1 => {
                        self.increase_max_hp(5);
                        self.event_damage(e.n[0], false, false);
                    }
                    _ => self.obtain_new(CardId::Decay),
                }
                e.leave(1);
            }
            (Ghosts, _) => {
                if k == 0 {
                    self.decrease_max_hp(e.n[0]);
                    for _ in 0..if asc15 { 3 } else { 5 } {
                        self.obtain_new(CardId::Apparition);
                    }
                }
                e.leave(1);
            }
            (MaskedBandits, 0) => {
                if k == 0 {
                    self.event_lose_gold(self.gold);
                    e.screen = 1;
                    e.set(&[("continue", true)]);
                } else {
                    let gold = self.rng.misc.random_range(25, 35);
                    let r = if self.has_relic(RelicId::RedMask) { RelicId::Circlet } else { RelicId::RedMask };
                    self.event_combat("Masked Bandits", vec![Reward::Gold(gold), Reward::Relic(r)], false);
                    return Ok(());
                }
            }
            (MaskedBandits, 1) => {
                e.screen = 2;
                e.set(&[("continue", true)]);
            }
            (MaskedBandits, _) => e.leave(3),
            (Nest, 0) => {
                e.screen = 1;
                let smash = format!("smash and grab: {} gold", e.n[0]);
                e.set(&[(&smash, true), ("stay in line: 6 damage, Ritual Dagger", true)]);
            }
            (Nest, _) => {
                if k == 0 {
                    self.gain_gold(e.n[0]);
                } else {
                    self.event_damage(6, false, false);
                    if matches!(self.screen, Screen::Over) {
                        return Ok(());
                    }
                    self.obtain_new(CardId::RitualDagger);
                }
                e.leave(2);
            }
            (TheLibrary, 0) => {
                if k == 1 {
                    self.heal(e.n[0]);
                    e.leave(2);
                } else {
                    let mut cards: Vec<crate::cards::Card> = Vec::new();
                    while cards.len() < 20 {
                        let mut id = self.library_card();
                        while cards.iter().any(|c| c.id == id) {
                            id = self.library_card();
                        }
                        let mut c = self.new_card(id);
                        self.preview_obtain(&mut c);
                        // `group.addToBottom(card)`: index 0, so the grid lists them last first, and
                        // a pick's index is the grid's (it took the 20th card back where the game took
                        // the 1st: soak 2026-10-27, run GJL5).
                        cards.insert(0, c);
                    }
                    let labels: Vec<String> = cards.iter().map(|c| format!("take {}", c.name())).collect();
                    let opts: Vec<(&str, bool)> = labels.iter().map(|l| (l.as_str(), true)).collect();
                    e.set(&opts);
                    e.cards = cards;
                    e.screen = 1;
                }
            }
            (TheLibrary, _) => {
                // The grid's pick is copied with makeCopy (a fresh, unupgraded card).
                let id = e.cards[k].id;
                self.obtain_new(id);
                e.cards.clear();
                e.leave(2);
            }
            (TheMausoleum, _) => {
                if k == 0 {
                    let cursed = self.rng.misc.random_boolean() || asc15;
                    let tier = self.random_relic_tier();
                    let r = self.random_screenless_relic(tier);
                    if cursed {
                        let writhe = self.new_card(CardId::Writhe);
                        self.obtain_card_then_relic(writhe, r);
                    } else {
                        self.obtain_relic(r);
                    }
                }
                e.leave(1);
            }
            (Vampires, _) => {
                let vial = e.n[1] == 1;
                let accept = k == 0 || (vial && k == 1);
                if accept {
                    if k == 0 {
                        self.decrease_max_hp(e.n[0]);
                    } else {
                        self.lose_relic(RelicId::BloodVial);
                    }
                    let strikes: Vec<usize> = (0..self.deck.len()).filter(|&i| self.deck[i].data().starter_strike).collect();
                    self.take_cards(&strikes);
                    self.on_master_deck_change();
                    for _ in 0..5 {
                        self.obtain_new(CardId::Bite);
                    }
                }
                e.leave(1);
            }
            (Designer, 0) => {
                let (upgrade_one, removes) = (e.n[0] == 1, e.n[1] == 1);
                let cost = e.list.clone();
                let up = !self.upgradable().is_empty();
                let nb = self.non_bottled();
                let adjust = if upgrade_one { "adjustments: upgrade a card" } else { "adjustments: upgrade 2 random cards" };
                let clean = if removes { "clean up: remove a card" } else { "clean up: transform 2 cards" };
                let adjust = format!("{adjust} ({} gold)", cost[0]);
                let clean = format!("{clean} ({} gold)", cost[1]);
                let full = format!("full service: remove a card, upgrade a random card ({} gold)", cost[2]);
                let punch = format!("punch: lose {} HP", cost[3]);
                let clean_ok = self.gold >= cost[1] as i32 && if removes { nb > 0 } else { nb >= 2 };
                e.set(&[
                    (&adjust, self.gold >= cost[0] as i32 && up),
                    (&clean, clean_ok),
                    (&full, self.gold >= cost[2] as i32 && nb > 0),
                    (&punch, true),
                ]);
                e.screen = 1;
            }
            (Designer, _) => {
                let cost = e.list.clone();
                e.leave(2);
                match k {
                    0 => {
                        self.event_lose_gold(cost[0] as i32);
                        if e.n[0] == 1 {
                            let opts = self.upgradable();
                            self.event_grid(e, GridPurpose::Upgrade, opts, 1);
                            return Ok(());
                        }
                        self.upgrade_random_shuffled(2);
                    }
                    1 => {
                        self.event_lose_gold(cost[1] as i32);
                        let opts = self.purgeable_unbottled();
                        if e.n[1] == 1 {
                            self.event_grid(e, GridPurpose::Remove, opts, 1);
                        } else {
                            self.event_grid(e, GridPurpose::EventTransform, opts, 2);
                        }
                        return Ok(());
                    }
                    2 => {
                        self.event_lose_gold(cost[2] as i32);
                        let opts = self.purgeable_unbottled();
                        self.event_grid(e, GridPurpose::RemoveAndUpgrade, opts, 1);
                        return Ok(());
                    }
                    _ => self.event_damage(cost[3] as i32, false, true),
                }
            }
            (Duplicator, _) => {
                e.leave(2);
                if k == 0 {
                    let opts: Vec<usize> = (0..self.deck.len()).collect();
                    self.event_grid(e, GridPurpose::Duplicate, opts, 1);
                    return Ok(());
                }
            }
            (KnowingSkull, 0) => {
                e.screen = 1;
                self.skull_options(&mut e);
            }
            (KnowingSkull, _) => {
                let cost = e.n[k.min(3)];
                self.event_damage(cost, false, true);
                if matches!(self.screen, Screen::Over) {
                    return Ok(());
                }
                match k {
                    0 => {
                        if !self.has_relic(RelicId::Sozu) {
                            let p = crate::run::pools::random_potion(self.config.character, &mut self.rng.potion);
                            self.obtain_potion(p);
                        }
                    }
                    1 => self.gain_gold(90),
                    2 => {
                        let id = self.return_colorless_card(CardRarity::Uncommon);
                        self.obtain_new(id);
                    }
                    _ => {
                        e.leave(2);
                        self.show(e);
                        return Ok(());
                    }
                }
                e.n[k] += 1;
                self.skull_options(&mut e);
            }
            (Nloth, _) => {
                if k < 2 {
                    if self.has_relic(RelicId::NlothsGift) {
                        self.obtain_relic(RelicId::Circlet);
                    } else {
                        if let Some(&r) = e.relics.get(k) {
                            self.lose_relic(r);
                        }
                        self.obtain_relic(RelicId::NlothsGift);
                    }
                }
                e.leave(1);
            }
            (TheJoust, 0) => {
                e.screen = 1;
                e.set(&[("bet on the murderer: 50 gold to win 100", true), ("bet on the owner: 50 gold to win 250", true)]);
            }
            (TheJoust, 1) => {
                e.n[0] = k as i32; // 1 = bet on the owner
                self.event_lose_gold(50);
                e.screen = 2;
                e.set(&[("watch", true)]);
            }
            (TheJoust, 2) => {
                e.n[1] = self.rng.misc.random_boolean_chance(0.3) as i32;
                e.screen = 3;
                e.set(&[("continue", true)]);
            }
            (TheJoust, _) => {
                let owner_wins = e.n[1] == 1;
                let bet_owner = e.n[0] == 1;
                if owner_wins && bet_owner {
                    self.gain_gold(250);
                } else if !owner_wins && !bet_owner {
                    self.gain_gold(100);
                }
                e.leave(4);
            }
            _ => return self.beyond_event_choose(e, k),
        }
        if !matches!(self.screen, Screen::Over) {
            self.show(e);
        }
        Ok(())
    }

    fn skull_options(&self, e: &mut Event) {
        let potion = format!("a potion (lose {} HP)", e.n[0]);
        let gold = format!("90 gold (lose {} HP)", e.n[1]);
        let card = format!("a colorless card (lose {} HP)", e.n[2]);
        let leave = format!("leave (lose {} HP)", e.n[3]);
        e.set(&[(&potion, true), (&gold, true), (&card, true), (&leave, true)]);
    }

    /// `getCard(rollRarity())` for The Library.
    pub(crate) fn library_card(&mut self) -> CardId {
        let rarity = self.roll_card_rarity(super::super::pools::RewardRoom::Normal);
        self.pools.random_card(rarity, &mut self.rng.card)
    }
}
