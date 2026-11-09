//! Relic effects outside combat: `onEquip`, card obtain hooks, room entry, campfire options
//! and chests.

use sts_rng::{java_shuffle, JavaRandom};

use super::map::RoomType;
use super::pools::{self, RewardRoom};
use super::{GridPurpose, Reward, Run, Screen};
use crate::cards::{Card, CardColor, CardId, CardType};
use crate::relics::{RelicId, RelicState, RelicTier};

impl Run {
    /// Opens a sub-screen (a grid or reward list) from the current one, which is resumed after.
    pub(super) fn push_screen(&mut self, screen: Screen) {
        let current = std::mem::replace(&mut self.screen, screen);
        self.screen_stack.push(current);
    }

    /// Leaves a sub-screen: back to the screen it was opened from, or the map.
    pub(super) fn pop_screen(&mut self) {
        match self.screen_stack.pop() {
            Some(Screen::Over) => self.finish_act(),
            Some(s) => self.screen = s,
            None => self.screen = Screen::Map,
        }
    }

    /// Obtaining a relic: `instantObtain` / `obtain` and then `onEquip`.
    pub fn obtain_relic(&mut self, id: RelicId) {
        if id == RelicId::Circlet {
            if let Some(c) = self.relics.iter_mut().find(|r| r.id == RelicId::Circlet) {
                c.counter += 1;
                return;
            }
        }
        // Starter-relic upgrades take the starter's slot (BossRelicSelectScreen).
        if let Some(starter) = id.replaces_starter() {
            if let Some(i) = self.relics.iter().position(|r| r.id == starter) {
                self.relics[i] = RelicState::new(id);
                self.on_equip(id);
                return;
            }
        }
        self.relics.push(RelicState::new(id));
        self.on_equip(id);
    }

    /// The current room's card reward odds (`getCurrRoom().getCardRarity`).
    fn reward_room_here(&self) -> RewardRoom {
        match self.room {
            Some(RoomType::Shop) => RewardRoom::Shop,
            Some(RoomType::Elite) => RewardRoom::Elite,
            _ => RewardRoom::Normal,
        }
    }

    fn relic_mut(&mut self, id: RelicId) -> Option<&mut RelicState> {
        self.relics.iter_mut().find(|r| r.id == id)
    }

    fn purgeable(&self) -> Vec<usize> {
        super::neow::purgeable(self)
    }

    /// Upgrades `n` random upgradable cards picked by a miscRng shuffle (War Paint, Whetstone,
    /// Tiny House).
    fn upgrade_random(&mut self, n: usize, t: Option<CardType>) {
        let mut cands: Vec<usize> = (0..self.deck.len()).filter(|&i| self.deck[i].can_upgrade() && t.map_or(true, |t| self.deck[i].card_type() == t)).collect();
        let seed = self.rng.misc.random_long();
        java_shuffle(&mut cands, &mut JavaRandom::new(seed));
        for &i in cands.iter().take(n) {
            self.deck[i].upgrade();
        }
    }

    fn on_equip(&mut self, id: RelicId) {
        use RelicId::*;
        match id {
            Astrolabe => {
                let opts = self.purgeable();
                if opts.len() <= 3 {
                    if !opts.is_empty() {
                        self.astrolabe(&opts);
                    }
                } else {
                    self.push_screen(Screen::Grid { purpose: GridPurpose::Astrolabe, options: opts, count: 3 });
                }
            }
            BottledFlame | BottledLightning | BottledTornado => {
                let want = match id {
                    BottledFlame => CardType::Attack,
                    BottledLightning => CardType::Skill,
                    _ => CardType::Power,
                };
                let opts: Vec<usize> = self.purgeable().into_iter().filter(|&i| self.deck[i].card_type() == want).collect();
                if !opts.is_empty() {
                    self.push_screen(Screen::Grid { purpose: GridPurpose::Bottle(id), options: opts, count: 1 });
                }
            }
            CallingBell => {
                let curse = self.new_card(CardId::CurseOfTheBell);
                self.obtain_card(curse);
                // combatRewardScreen.open() rolls a card reward before the list is replaced.
                let _ = self.reward_cards(RewardRoom::Normal);
                let items = vec![
                    Reward::Relic(self.random_screenless_relic(RelicTier::Common)),
                    Reward::Relic(self.random_screenless_relic(RelicTier::Uncommon)),
                    Reward::Relic(self.random_screenless_relic(RelicTier::Rare)),
                ];
                self.push_screen(Screen::Rewards { items });
            }
            Cauldron => {
                let items: Vec<Reward> = (0..5).map(|_| Reward::Potion(super::pools::random_potion(self.config.character, &mut self.rng.potion))).collect();
                // open() rolls a card reward, which Cauldron then removes.
                let room = self.reward_room_here();
                let _ = self.reward_cards(room);
                self.labelled_rewards = true;
                self.push_screen(Screen::Rewards { items });
            }
            Orrery => {
                // Four card rewards, plus the one combatRewardScreen.open() adds.
                let room = self.reward_room_here();
                let items: Vec<Reward> = (0..5).map(|_| Reward::Cards(self.reward_cards(room))).collect();
                self.labelled_rewards = true;
                self.push_screen(Screen::Rewards { items });
            }
            DollysMirror => {
                let opts: Vec<usize> = (0..self.deck.len()).collect();
                if !opts.is_empty() {
                    self.push_screen(Screen::Grid { purpose: GridPurpose::Duplicate, options: opts, count: 1 });
                }
            }
            DuVuDoll => self.recount_curses(),
            Necronomicon => {
                // ShowCardAndObtainEffect: Omamori can cancel it, and the obtain reactions run.
                let curse = self.new_card(CardId::Necronomicurse);
                self.obtain_card(curse);
            }
            EmptyCage => {
                let opts = self.purgeable();
                if opts.len() <= 2 {
                    let mut idx = opts;
                    idx.sort_unstable();
                    for i in idx.into_iter().rev() {
                        self.deck.remove(i);
                    }
                    self.on_master_deck_change();
                } else {
                    self.push_screen(Screen::Grid { purpose: GridPurpose::Remove, options: opts, count: 2 });
                }
            }
            HappyFlower | IncenseBurner | Sundial | TinyChest => self.relic_mut(id).unwrap().counter = 0,
            Mango => self.increase_max_hp(14),
            Pear => self.increase_max_hp(10),
            Strawberry => self.increase_max_hp(7),
            Waffle => {
                self.max_hp += 7;
                let m = self.max_hp;
                self.heal(m);
            }
            OldCoin => self.gain_gold(300),
            PandorasBox => {
                let mut count = 0;
                let mut keep = Vec::new();
                for c in std::mem::take(&mut self.deck) {
                    if c.data().starter_strike || c.data().starter_defend {
                        count += 1;
                    } else {
                        keep.push(c);
                    }
                }
                self.deck = keep;
                let mut cards = Vec::new();
                for _ in 0..count {
                    let id = self.truly_random_card();
                    let mut c = self.new_card(id);
                    self.preview_obtain(&mut c);
                    cards.push(c);
                }
                // The game adds them to its confirmation grid with `addToBottom` (index 0), then
                // obtains them in grid order: the last card rolled joins the deck first.
                for c in cards.into_iter().rev() {
                    self.obtain_card(c);
                }
            }
            PotionBelt => {
                self.potions.push(None);
                self.potions.push(None);
            }
            TinyHouse => {
                self.upgrade_random(1, None);
                self.increase_max_hp(5);
                let potion = pools::random_potion(self.config.character, &mut self.rng.misc);
                let mut items = vec![Reward::Gold(50), Reward::Potion(potion)];
                let cards = self.reward_cards(RewardRoom::Normal);
                if !cards.is_empty() {
                    items.push(Reward::Cards(cards));
                }
                self.labelled_rewards = true;
                self.push_screen(Screen::Rewards { items });
            }
            WarPaint => self.upgrade_random(2, Some(CardType::Skill)),
            Whetstone => self.upgrade_random(2, Some(CardType::Attack)),
            FrozenEgg2 | MoltenEgg2 | ToxicEgg2 => {
                // Upgrade the cards already on the open reward screen.
                let mut rewards = self.saved_rewards.take();
                for list in [rewards.as_mut(), self.screen_rewards_mut()].into_iter().flatten() {
                    for r in list.iter_mut() {
                        if let Reward::Cards(cs) = r {
                            for c in cs.iter_mut() {
                                egg_upgrade(id, c);
                            }
                        }
                    }
                }
                self.saved_rewards = rewards;
            }
            _ => {}
        }
    }

    fn screen_rewards_mut(&mut self) -> Option<&mut Vec<Reward>> {
        match &mut self.screen {
            Screen::Rewards { items } => Some(items),
            _ => self.screen_stack.iter_mut().rev().find_map(|s| match s {
                Screen::Rewards { items } => Some(items),
                _ => None,
            }),
        }
    }

    /// Astrolabe's `giveCards`: transform each card with miscRng and upgrade the result.
    pub(super) fn astrolabe(&mut self, indices: &[usize]) {
        let removed = self.take_cards(indices);
        for c in removed {
            let id = super::neow::transformed_id(self, c.id, TransformRng::Misc);
            let mut card = self.new_card(id);
            if card.can_upgrade() {
                card.upgrade();
            }
            self.obtain_card(card);
        }
    }

    /// Removes the deck cards at `indices`, returning them in the given order.
    pub(super) fn take_cards(&mut self, indices: &[usize]) -> Vec<Card> {
        let picked: Vec<Card> = indices.iter().map(|&i| self.deck[i].clone()).collect();
        let mut idx = indices.to_vec();
        idx.sort_unstable();
        for i in idx.into_iter().rev() {
            self.deck.remove(i);
        }
        // onRemoveFromMasterDeck: Parasite.
        for c in &picked {
            if c.id == CardId::Parasite {
                self.decrease_max_hp(3);
            }
        }
        picked
    }

    /// `returnTrulyRandomCard()`: any card of the src pools, with cardRandomRng.
    fn truly_random_card(&mut self) -> CardId {
        let mut list = Vec::new();
        for r in [crate::cards::CardRarity::Common, crate::cards::CardRarity::Uncommon, crate::cards::CardRarity::Rare] {
            list.extend(self.pools.src_pool(r));
        }
        list[self.rng.card_random.random(list.len() as i32 - 1) as usize]
    }

    /// `returnRandomScreenlessRelic(tier)`.
    pub(super) fn random_screenless_relic(&mut self, tier: RelicTier) -> RelicId {
        loop {
            let r = self.random_relic(tier);
            if !matches!(r, RelicId::BottledFlame | RelicId::BottledLightning | RelicId::BottledTornado | RelicId::Whetstone) {
                return r;
            }
        }
    }

    /// For training's practice fights (not the game): a random relic of `tier` from the run's
    /// pools whose pickup opens no screen (no card grid or reward list to settle first), so a run
    /// set up partway through an act can hold any of the rest.
    pub fn random_relic_no_screen(&mut self, tier: RelicTier) -> RelicId {
        use RelicId::*;
        loop {
            let r = self.random_relic(tier);
            if !matches!(r, Astrolabe | BottledFlame | BottledLightning | BottledTornado | CallingBell | Cauldron | DollysMirror | EmptyCage | Orrery | TinyHouse) {
                return r;
            }
        }
    }

    /// `AbstractDungeon.returnRandomRelicTier()` (digging, events): the act's chances,
    /// `commonRelicChance` and `uncommonRelicChance`, 50 and 33 in Acts 1-3 and 0 and 100 in The
    /// Ending, so Act 4's are always uncommon (the roll is made all the same). Elite and monster
    /// rooms roll their own, always 50/33/17 (`reward_screen`). (It rolled Acts 1-3's everywhere:
    /// a dig at Act 4's rest site gave a Captain's Wheel where the game gave a Mummified Hand,
    /// soak 2026-10-27, run GBFJ.)
    pub(super) fn random_relic_tier(&mut self) -> RelicTier {
        let roll = self.rng.relic.random_range(0, 99);
        let (common, uncommon) = if self.act == 4 { (0, 100) } else { (50, 33) };
        if roll < common {
            RelicTier::Common
        } else if roll < common + uncommon {
            RelicTier::Uncommon
        } else {
            RelicTier::Rare
        }
    }

    // ---- Cards ----

    /// `onPreviewObtainCard`: the eggs upgrade matching cards shown as rewards.
    pub(super) fn preview_obtain(&self, c: &mut Card) {
        for r in &self.relics {
            egg_upgrade(r.id, c);
        }
    }

    /// `ShowCardAndObtainEffect`: Omamori may cancel a curse; otherwise `onObtainCard`, the
    /// card joins the deck, then `onMasterDeckChange`.
    pub fn obtain_card(&mut self, card: Card) {
        if !self.omamori_cancels(&card) {
            self.finish_obtain_card(card);
        }
    }

    /// A card's `ShowCardAndObtainEffect` started in the same click as a relic is obtained, card
    /// first (Addict's rob, The Mausoleum, Big Fish's box): Omamori is checked as the effect
    /// starts, before the relic arrives (a robbed Omamori doesn't stop that curse); the card's
    /// `onObtainCard` and its joining the deck come when the effect ends, after the relic (a new
    /// Ceramic Fish or Darkstone Periapt does react to it).
    pub(super) fn obtain_card_then_relic(&mut self, card: Card, relic: RelicId) {
        let cancelled = self.omamori_cancels(&card);
        self.obtain_relic(relic);
        if !cancelled {
            self.finish_obtain_card(card);
        }
    }

    /// `ShowCardAndObtainEffect`'s constructor: Omamori cancels a curse (using a charge).
    fn omamori_cancels(&mut self, card: &Card) -> bool {
        if card.data().color == CardColor::Curse {
            if let Some(o) = self.relic_mut(RelicId::Omamori) {
                if o.counter != 0 {
                    o.counter -= 1;
                    if o.counter == 0 {
                        o.used_up = true;
                    }
                    return true;
                }
            }
        }
        false
    }

    /// The end of `ShowCardAndObtainEffect`: `onObtainCard`, the card joins the deck, then
    /// `onMasterDeckChange`.
    fn finish_obtain_card(&mut self, mut card: Card) {
        let ids: Vec<RelicId> = self.relics.iter().map(|r| r.id).collect();
        for id in ids {
            match id {
                RelicId::CeramicFish => self.gain_gold(9),
                RelicId::DarkstonePeriapt => {
                    if card.data().color == CardColor::Curse {
                        self.increase_max_hp(6);
                    }
                }
                RelicId::FrozenEgg2 | RelicId::MoltenEgg2 | RelicId::ToxicEgg2 => egg_upgrade(id, &mut card),
                _ => {}
            }
        }
        self.deck.push(card);
        self.on_master_deck_change();
    }

    /// A card obtained during a fight (`Combat::obtain_card_in_combat`, which already ran Omamori,
    /// Darkstone Periapt and Ceramic Fish) joining the deck when the fight ends: only the eggs'
    /// upgrade is left of `onObtainCard`.
    pub(super) fn add_obtained_card(&mut self, mut card: Card) {
        let ids: Vec<RelicId> = self.relics.iter().map(|r| r.id).collect();
        for id in ids {
            if matches!(id, RelicId::FrozenEgg2 | RelicId::MoltenEgg2 | RelicId::ToxicEgg2) {
                egg_upgrade(id, &mut card);
            }
        }
        self.deck.push(card);
        self.on_master_deck_change();
    }

    pub(super) fn on_master_deck_change(&mut self) {
        self.recount_curses();
    }

    fn recount_curses(&mut self) {
        let n = self.deck.iter().filter(|c| c.card_type() == CardType::Curse).count() as i32;
        if let Some(r) = self.relic_mut(RelicId::DuVuDoll) {
            r.counter = n;
        }
    }

    // ---- Rooms ----

    /// `onEnterRoom` (the room before the ? roll).
    pub(super) fn relics_on_enter_room(&mut self, room: RoomType) {
        let ids: Vec<RelicId> = self.relics.iter().map(|r| r.id).collect();
        for id in ids {
            match id {
                RelicId::MawBank => {
                    if !self.relic_mut(id).unwrap().used_up {
                        self.gain_gold(12);
                    }
                }
                RelicId::EternalFeather => {
                    if room == RoomType::Rest {
                        let amt = self.deck.len() as i32 / 5 * 3;
                        self.heal(amt);
                    }
                }
                RelicId::SsserpentHead => {
                    if room == RoomType::Event {
                        self.gain_gold(50);
                    }
                }
                _ => {}
            }
        }
    }

    /// `justEnteredRoom` (after the ? roll).
    pub(super) fn relics_just_entered_room(&mut self, room: RoomType) {
        if room == RoomType::Shop && self.has_relic(RelicId::MealTicket) {
            self.heal(15);
        }
    }

    /// `RestRoom.onPlayerEntry`: `onEnterRestRoom`.
    pub(super) fn relics_on_enter_rest_room(&mut self) {
        if let Some(r) = self.relic_mut(RelicId::AncientTeaSet) {
            r.counter = -2;
        }
    }

    /// `AbstractChest.open`: `onChestOpen` adds to the rewards before the chest's own.
    pub(super) fn relics_on_chest_open(&mut self, items: &mut Vec<Reward>) {
        let ids: Vec<RelicId> = self.relics.iter().map(|r| r.id).collect();
        for id in ids {
            match id {
                RelicId::CursedKey => {
                    let curse = super::neow::random_curse(&mut self.rng.card);
                    let c = self.new_card(curse);
                    self.obtain_card(c);
                }
                RelicId::Matryoshka => {
                    if self.relic_mut(id).unwrap().counter > 0 {
                        let r = self.relic_mut(id).unwrap();
                        r.counter -= 1;
                        let tier = if self.rng.relic.random_boolean_chance(0.75) { RelicTier::Common } else { RelicTier::Uncommon };
                        if self.relic_mut(id).unwrap().counter == 0 {
                            let r = self.relic_mut(id).unwrap();
                            r.counter = -2;
                            r.used_up = true;
                        }
                        let relic = self.random_relic(tier);
                        items.push(Reward::Relic(relic));
                    }
                }
                _ => {}
            }
        }
    }

    /// `onChestOpenAfter`: N'loth's Mask takes one relic back.
    pub(super) fn relics_on_chest_open_after(&mut self, items: &mut Vec<Reward>) {
        if let Some(r) = self.relic_mut(RelicId::NlothsMask) {
            if r.counter > 0 {
                r.counter -= 1;
                if r.counter == 0 {
                    r.counter = -2;
                    r.used_up = true;
                }
                // removeOneRelicFromRewards: the relic, and the sapphire key linked to it.
                if let Some(i) = items.iter().position(|x| matches!(x, Reward::Relic(_))) {
                    items.remove(i);
                    if items.get(i) == Some(&Reward::SapphireKey) {
                        items.remove(i);
                    }
                }
            }
        }
    }
}

/// Which stream a transform draws from.
#[derive(Clone, Copy)]
pub(super) enum TransformRng {
    Neow,
    Misc,
}

fn egg_upgrade(id: RelicId, c: &mut Card) {
    let t = match id {
        RelicId::FrozenEgg2 => CardType::Power,
        RelicId::MoltenEgg2 => CardType::Attack,
        RelicId::ToxicEgg2 => CardType::Skill,
        _ => return,
    };
    if c.card_type() == t && c.can_upgrade() && !c.upgraded {
        c.upgrade();
    }
}
