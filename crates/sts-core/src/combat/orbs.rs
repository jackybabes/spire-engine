//! The Defect's orbs (`AbstractOrb` and its four kinds) and the player's orb slots
//! (`AbstractPlayer.channelOrb`, `evokeOrb`, `increaseMaxOrbSlots`...).
//!
//! An orb's numbers follow Focus at all times (the game's orbs call `applyFocus` every frame),
//! so only what can't be recomputed is stored: a Dark orb's evoke damage, which grows each turn.

use serde::{Deserialize, Serialize};

use super::Combat;
use crate::action::{Action, Cr, DamageInfo, DamageSource, DamageType};
use crate::powers::PowerId;
use crate::relics::RelicId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrbId {
    Lightning,
    Frost,
    Dark,
    Plasma,
}

impl OrbId {
    /// `AbstractOrb.ID`.
    pub fn java_id(self) -> &'static str {
        match self {
            OrbId::Lightning => "Lightning",
            OrbId::Frost => "Frost",
            OrbId::Dark => "Dark",
            OrbId::Plasma => "Plasma",
        }
    }

    fn base_passive(self) -> i32 {
        match self {
            OrbId::Lightning => 3,
            OrbId::Frost => 2,
            OrbId::Dark => 6,
            OrbId::Plasma => 1,
        }
    }

    fn base_evoke(self) -> i32 {
        match self {
            OrbId::Lightning => 8,
            OrbId::Frost => 5,
            OrbId::Dark => 6,
            OrbId::Plasma => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Orb {
    pub id: OrbId,
    /// A Dark orb's `evokeAmount` (Focus doesn't change it; each end of turn adds the passive).
    pub dark_evoke: i32,
}

impl Orb {
    pub fn new(id: OrbId) -> Orb {
        Orb { id, dark_evoke: id.base_evoke() }
    }
}

/// `AbstractOrb.getRandomOrb(true)`: the list's order is Dark, Frost, Lightning, Plasma.
pub(crate) const RANDOM_ORBS: [OrbId; 4] = [OrbId::Dark, OrbId::Frost, OrbId::Lightning, OrbId::Plasma];

impl Combat {
    fn focus(&self) -> Option<i32> {
        self.player.power(PowerId::Focus).map(|p| p.amount)
    }

    /// `passiveAmount` after `applyFocus` (Plasma ignores Focus).
    pub fn orb_passive(&self, o: &Orb) -> i32 {
        match self.focus() {
            Some(f) if o.id != OrbId::Plasma => (o.id.base_passive() + f).max(0),
            _ => o.id.base_passive(),
        }
    }

    /// `evokeAmount` after `applyFocus` (Plasma ignores Focus; Dark keeps its own).
    pub fn orb_evoke(&self, o: &Orb) -> i32 {
        match (o.id, self.focus()) {
            (OrbId::Dark, _) => o.dark_evoke,
            (OrbId::Plasma, _) | (_, None) => o.id.base_evoke(),
            (_, Some(f)) => (o.id.base_evoke() + f).max(0),
        }
    }

    pub fn filled_orb_count(&self) -> usize {
        self.player.orbs.iter().filter(|o| o.is_some()).count()
    }

    /// `hasOrb`: the first slot is filled.
    pub fn has_orb(&self) -> bool {
        matches!(self.player.orbs.first(), Some(Some(_)))
    }

    pub fn has_empty_orb(&self) -> bool {
        self.player.orbs.iter().any(|o| o.is_none())
    }

    /// `AbstractPlayer.channelOrb`: the first empty slot, or evoke the front orb first.
    pub(crate) fn channel_orb(&mut self, orb: Orb) {
        if self.player.orbs.is_empty() {
            return;
        }
        if let Some(i) = self.player.orbs.iter().position(|o| o.is_none()) {
            self.player.orbs[i] = Some(orb);
            self.orbs_channeled_this_combat.push(orb.id);
            self.orbs_channeled_this_turn.push(orb.id);
        } else {
            // addToTop in this order: the evoke runs first, then the channel.
            self.push_top(Action::Channel { orb, auto_evoke: true });
            self.push_top(Action::EvokeOrb { count: 1 });
        }
    }

    /// `AbstractOrb.onEvoke` for an orb with these numbers.
    fn on_evoke(&mut self, o: &Orb) {
        let n = self.orb_evoke(o);
        match o.id {
            OrbId::Lightning => {
                let hit_all = self.player.has_power(PowerId::Electro);
                self.push_top(Action::LightningOrbDamage { amount: n, hit_all });
            }
            OrbId::Frost => self.push_top(Action::GainBlock { target: Cr::Player, amount: n }),
            OrbId::Dark => {
                // DarkOrbEvokeAction picks the monster with the least HP when it is made.
                let mut weakest: Option<usize> = None;
                for &m in &self.order {
                    if !self.monsters[m].is_dead_or_escaped() && weakest.map_or(true, |w| self.monsters[m].hp < self.monsters[w].hp) {
                        weakest = Some(m);
                    }
                }
                self.push_top(Action::DarkOrbEvoke { target: weakest, amount: n });
            }
            OrbId::Plasma => self.push_top(Action::GainEnergy { amount: n }),
        }
    }

    /// `evokeOrb`: the front orb goes off and the rest move up.
    pub(crate) fn evoke_orb(&mut self) {
        if let Some(Some(o)) = self.player.orbs.first().copied() {
            self.on_evoke(&o);
            self.remove_next_orb();
        }
    }

    /// `evokeWithoutLosingOrb`.
    pub(crate) fn evoke_without_losing_orb(&mut self) {
        if let Some(Some(o)) = self.player.orbs.first().copied() {
            self.on_evoke(&o);
        }
    }

    /// `removeNextOrb`: drop the front orb without evoking it.
    pub(crate) fn remove_next_orb(&mut self) {
        if matches!(self.player.orbs.first(), Some(Some(_))) {
            self.player.orbs.remove(0);
            self.player.orbs.push(None);
        }
    }

    /// `increaseMaxOrbSlots` (at most 10 slots; a call that starts at 10 does nothing).
    pub(crate) fn increase_max_orb_slots(&mut self, amount: usize) {
        if self.player.orbs.len() == 10 {
            return;
        }
        for _ in 0..amount {
            self.player.orbs.push(None);
        }
    }

    /// `decreaseMaxOrbSlots(1)`: the last slot goes, orb and all.
    pub(crate) fn decrease_max_orb_slots(&mut self) {
        self.player.orbs.pop();
    }

    /// `AbstractOrb.onStartOfTurn` (only Plasma does anything).
    fn orb_start_of_turn(&mut self, o: &Orb) {
        if o.id == OrbId::Plasma {
            let amount = self.orb_passive(o);
            self.push_bot(Action::GainEnergy { amount });
        }
    }

    /// `AbstractOrb.onEndOfTurn` for the orb in slot `i`.
    fn orb_end_of_turn(&mut self, i: usize) {
        let Some(Some(o)) = self.player.orbs.get(i).copied() else { return };
        let amount = self.orb_passive(&o);
        match o.id {
            OrbId::Lightning => {
                let hit_all = self.player.has_power(PowerId::Electro);
                self.push_bot(Action::LightningOrbDamage { amount, hit_all });
            }
            OrbId::Frost => self.push_bot(Action::GainBlock { target: Cr::Player, amount }),
            OrbId::Dark => {
                if let Some(Some(d)) = self.player.orbs.get_mut(i) {
                    d.dark_evoke += amount;
                }
            }
            OrbId::Plasma => {}
        }
    }

    /// `applyStartOfTurnOrbs`: every orb, then the front one again with Gold-Plated Cables.
    pub(crate) fn apply_start_of_turn_orbs(&mut self) {
        for o in self.player.orbs.clone().into_iter().flatten() {
            self.orb_start_of_turn(&o);
        }
        if self.has_relic(RelicId::GoldPlatedCables) {
            if let Some(Some(o)) = self.player.orbs.first().copied() {
                self.orb_start_of_turn(&o);
            }
        }
    }

    /// `TriggerEndOfTurnOrbsAction`: every orb's end-of-turn effect, the front one twice with
    /// Gold-Plated Cables.
    pub(crate) fn trigger_end_of_turn_orbs(&mut self) {
        for i in 0..self.player.orbs.len() {
            self.orb_end_of_turn(i);
        }
        if self.has_relic(RelicId::GoldPlatedCables) && self.has_orb() {
            self.orb_end_of_turn(0);
        }
    }

    /// `ImpulseAction` (Emotion Chip) and `DarkImpulseAction` (Darkness+): each orb's start and
    /// end of turn effects, only Dark orbs' for the latter.
    pub(crate) fn impulse(&mut self, dark_only: bool) {
        for i in 0..self.player.orbs.len() {
            let Some(Some(o)) = self.player.orbs.get(i).copied() else { continue };
            if !dark_only || o.id == OrbId::Dark {
                self.orb_start_of_turn(&o);
                self.orb_end_of_turn(i);
            }
        }
        if self.has_relic(RelicId::GoldPlatedCables) {
            if let Some(Some(o)) = self.player.orbs.first().copied() {
                if !dark_only || o.id == OrbId::Dark {
                    self.orb_start_of_turn(&o);
                    self.orb_end_of_turn(0);
                }
            }
        }
    }

    /// `LoopPower.atStartOfTurn`: the front slot's start and end of turn effects.
    pub(crate) fn loop_front_orb(&mut self, times: i32) {
        if self.player.orbs.is_empty() {
            return;
        }
        for _ in 0..times {
            if let Some(Some(o)) = self.player.orbs.first().copied() {
                self.orb_start_of_turn(&o);
                self.orb_end_of_turn(0);
            }
        }
    }

    /// `AbstractOrb.applyLockOn`.
    pub(crate) fn lock_on(&self, m: usize, dmg: i32) -> i32 {
        if self.monsters[m].has_power(PowerId::LockOn) {
            (dmg as f32 * 1.5) as i32
        } else {
            dmg
        }
    }

    /// `LightningOrbEvokeAction` / `LightningOrbPassiveAction`: a random monster, or every one
    /// with Electrodynamics (`createDamageMatrix(base, true, true)`, Lock-On per monster).
    pub(crate) fn lightning_orb_damage(&mut self, amount: i32, hit_all: bool) {
        if !hit_all {
            if let Some(m) = self.get_random_monster(None, true, true) {
                let out = self.lock_on(m, amount);
                let info = DamageInfo::new(Some(Cr::Player), out, DamageType::Thorns);
                self.push_top(Action::Damage { target: Cr::Monster(m), source: DamageSource::Fixed(info), steal_gold: 0 });
            }
        } else {
            let damage: Vec<i32> = self.order.iter().map(|&m| self.lock_on(m, amount)).collect();
            self.push_top(Action::DamageAllEnemies { source: Some(Cr::Player), damage, damage_type: DamageType::Thorns });
        }
    }
}
