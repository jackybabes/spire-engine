//! `MapGenerator` + `RoomTypeAssigner` + `AbstractDungeon.generateMap`.

use serde::{Deserialize, Serialize};
use sts_rng::{java_shuffle, StsRandom};

/// Room types on the map. `Event` rooms are rolled into their real type on entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoomType {
    Monster,
    Elite,
    Event,
    Rest,
    Shop,
    Treasure,
    Boss,
}

impl RoomType {
    pub fn symbol(self) -> char {
        match self {
            RoomType::Monster => 'M',
            RoomType::Elite => 'E',
            RoomType::Event => '?',
            RoomType::Rest => 'R',
            RoomType::Shop => '$',
            RoomType::Treasure => 'T',
            RoomType::Boss => 'B',
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MapNode {
    pub x: i32,
    pub y: i32,
    pub room: Option<RoomType>,
    /// Outgoing edges as (dstX, dstY), kept sorted like `EdgeComparator`.
    pub edges: Vec<(i32, i32)>,
    /// Parent nodes as (x, y); duplicates are possible, as in the game.
    pub parents: Vec<(i32, i32)>,
    pub has_emerald_key: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Map {
    pub rows: Vec<Vec<MapNode>>,
}

pub const MAP_HEIGHT: usize = 15;
pub const MAP_WIDTH: usize = 7;
pub const PATH_DENSITY: usize = 6;

/// Chances from `Exordium.initializeLevelSpecificChances`.
pub struct RoomChances {
    pub shop: f32,
    pub rest: f32,
    pub treasure: f32,
    pub event: f32,
    pub elite: f32,
}

pub const EXORDIUM_CHANCES: RoomChances = RoomChances { shop: 0.05, rest: 0.12, treasure: 0.0, event: 0.22, elite: 0.08 };

/// `Math.round(float)`.
pub fn java_round(v: f32) -> i32 {
    (v + 0.5).floor() as i32
}

fn rand_range(rng: &mut StsRandom, min: i32, max: i32) -> i32 {
    rng.random(max - min) + min
}

impl Map {
    pub fn node(&self, x: i32, y: i32) -> &MapNode {
        &self.rows[y as usize][x as usize]
    }

    fn node_mut(&mut self, x: i32, y: i32) -> &mut MapNode {
        &mut self.rows[y as usize][x as usize]
    }

    fn add_edge(&mut self, x: i32, y: i32, e: (i32, i32)) {
        let n = self.node_mut(x, y);
        if !n.edges.contains(&e) {
            n.edges.push(e);
        }
        n.edges.sort();
    }

    /// `MapGenerator.generateDungeon(height, width, pathDensity, rng)`.
    pub fn generate_paths(rng: &mut StsRandom) -> Map {
        let rows = (0..MAP_HEIGHT as i32)
            .map(|y| {
                (0..MAP_WIDTH as i32)
                    .map(|x| MapNode { x, y, room: None, edges: Vec::new(), parents: Vec::new(), has_emerald_key: false })
                    .collect()
            })
            .collect();
        let mut map = Map { rows };
        let row_size = MAP_WIDTH as i32 - 1;
        let mut first = -1;
        for i in 0..PATH_DENSITY {
            let mut start = rand_range(rng, 0, row_size);
            if i == 0 {
                first = start;
            }
            while start == first && i == 1 {
                start = rand_range(rng, 0, row_size);
            }
            map.create_path(start, 0, rng);
        }
        map.filter_redundant_edges_from_first_row();
        map
    }

    /// `_createPaths` starting from the edge into (x, y).
    fn create_path(&mut self, mut x: i32, mut y: i32, rng: &mut StsRandom) {
        loop {
            if y + 1 >= MAP_HEIGHT as i32 {
                self.add_edge(x, y, (3, y + 2));
                return;
            }
            let row_end = MAP_WIDTH as i32 - 1;
            let (min, max) = if x == 0 {
                (0, 1)
            } else if x == row_end {
                (-1, 0)
            } else {
                (-1, 1)
            };
            let mut new_x = x + rand_range(rng, min, max);
            let new_y = y + 1;
            let mut target = (new_x, new_y);
            let parents = self.node(target.0, target.1).parents.clone();
            for parent in parents {
                if parent != (x, y) {
                    if let Some(ancestor) = self.common_ancestor(parent, (x, y), 5) {
                        let gap = new_y - ancestor.1;
                        if gap < 3 {
                            if target.0 > x {
                                new_x = x + rand_range(rng, -1, 0);
                                if new_x < 0 {
                                    new_x = x;
                                }
                            } else if target.0 == x {
                                new_x = x + rand_range(rng, -1, 1);
                                if new_x > row_end {
                                    new_x = x - 1;
                                } else if new_x < 0 {
                                    new_x = x + 1;
                                }
                            } else {
                                new_x = x + rand_range(rng, 0, 1);
                                if new_x > row_end {
                                    new_x = x;
                                }
                            }
                            target = (new_x, new_y);
                        }
                    }
                }
            }
            if x != 0 {
                let left = self.node(x - 1, y);
                if let Some(&(max_x, _)) = left.edges.iter().max() {
                    if max_x > new_x {
                        new_x = max_x;
                    }
                }
            }
            if x < row_end {
                let right = self.node(x + 1, y);
                if let Some(&(min_x, _)) = right.edges.iter().min() {
                    if min_x < new_x {
                        new_x = min_x;
                    }
                }
            }
            self.add_edge(x, y, (new_x, new_y));
            self.node_mut(new_x, new_y).parents.push((x, y));
            x = new_x;
            y = new_y;
        }
    }

    /// `getCommonAncestor`, including its `node1.x < node2.y` comparison.
    fn common_ancestor(&self, n1: (i32, i32), n2: (i32, i32), max_depth: i32) -> Option<(i32, i32)> {
        let (mut l, mut r) = if n1.0 < n2.1 { (n1, n2) } else { (n2, n1) };
        let mut cy = n1.1;
        while cy >= 0 && cy >= n1.1 - max_depth {
            let lp = &self.node(l.0, l.1).parents;
            let rp = &self.node(r.0, r.1).parents;
            if lp.is_empty() || rp.is_empty() {
                return None;
            }
            // getNodeWithMaxX / getNodeWithMinX keep the first node on ties.
            let mut lmax = lp[0];
            for &p in lp {
                if p.0 > lmax.0 {
                    lmax = p;
                }
            }
            let mut rmin = rp[0];
            for &p in rp {
                if p.0 < rmin.0 {
                    rmin = p;
                }
            }
            l = lmax;
            r = rmin;
            if l == r {
                return Some(l);
            }
            cy -= 1;
        }
        None
    }

    /// `filterRedundantEdgesFromRow` on row 0.
    fn filter_redundant_edges_from_first_row(&mut self) {
        let mut existing: Vec<(i32, i32)> = Vec::new();
        for x in 0..MAP_WIDTH {
            let edges = self.rows[0][x].edges.clone();
            if edges.is_empty() {
                continue;
            }
            let mut delete = Vec::new();
            for e in &edges {
                if existing.contains(e) {
                    delete.push(*e);
                }
                existing.push(*e);
            }
            self.rows[0][x].edges.retain(|e| !delete.contains(e));
        }
    }

    /// `TheEnding.generateSpecialMap()`: rest, shop, elite and boss in column 3, then the
    /// victory node.
    pub fn the_ending() -> Map {
        let rooms = [Some(RoomType::Rest), Some(RoomType::Shop), Some(RoomType::Elite), Some(RoomType::Boss), None];
        let mut rows = Vec::new();
        for (y, room) in rooms.iter().enumerate() {
            let y = y as i32;
            let row = (0..MAP_WIDTH as i32)
                .map(|x| {
                    let mut n = MapNode { x, y, room: None, edges: Vec::new(), parents: Vec::new(), has_emerald_key: false };
                    if x == 3 {
                        n.room = *room;
                        if y < 3 {
                            n.edges.push((3, y + 1));
                        }
                        if y > 0 {
                            n.parents.push((3, y - 1));
                        }
                    }
                    n
                })
                .collect();
            rows.push(row);
        }
        Map { rows }
    }

    /// `AbstractDungeon.generateMap()` for an act.
    pub fn generate(rng: &mut StsRandom, chances: &RoomChances, ascension: i32, emerald_key: bool) -> Map {
        let mut map = Map::generate_paths(rng);
        let count = map.rows.iter().flatten().filter(|n| !n.edges.is_empty() && n.y != MAP_HEIGHT as i32 - 2).count() as f32;
        let shop = java_round(count * chances.shop);
        let rest = java_round(count * chances.rest);
        let _treasure = java_round(count * chances.treasure);
        let elite = if ascension >= 1 { java_round(count * chances.elite * 1.6) } else { java_round(count * chances.elite) };
        let event = java_round(count * chances.event);
        let mut rooms: Vec<RoomType> = Vec::new();
        rooms.extend(std::iter::repeat(RoomType::Shop).take(shop.max(0) as usize));
        rooms.extend(std::iter::repeat(RoomType::Rest).take(rest.max(0) as usize));
        rooms.extend(std::iter::repeat(RoomType::Elite).take(elite.max(0) as usize));
        rooms.extend(std::iter::repeat(RoomType::Event).take(event.max(0) as usize));

        for n in map.rows[MAP_HEIGHT - 1].iter_mut() {
            n.room.get_or_insert(RoomType::Rest);
        }
        for n in map.rows[0].iter_mut() {
            n.room.get_or_insert(RoomType::Monster);
        }
        for n in map.rows[8].iter_mut() {
            n.room.get_or_insert(RoomType::Treasure);
        }
        map.distribute_rooms(rng, rooms);
        if emerald_key {
            let elites: Vec<(usize, usize)> = (0..map.rows.len())
                .flat_map(|y| (0..map.rows[y].len()).map(move |x| (y, x)))
                .filter(|&(y, x)| map.rows[y][x].room == Some(RoomType::Elite))
                .collect();
            if !elites.is_empty() {
                let k = rng.random_range(0, elites.len() as i32 - 1) as usize;
                let (y, x) = elites[k];
                map.rows[y][x].has_emerald_key = true;
            }
        }
        map
    }

    /// `RoomTypeAssigner.distributeRoomsAcrossMap`.
    fn distribute_rooms(&mut self, rng: &mut StsRandom, mut rooms: Vec<RoomType>) {
        let node_count = self.rows.iter().flatten().filter(|n| !n.edges.is_empty() && n.room.is_none()).count();
        while rooms.len() < node_count {
            rooms.push(RoomType::Monster);
        }
        // Shuffled with the RandomXS128 itself: the Random wrapper's counter does not move.
        java_shuffle(&mut rooms, &mut rng.rng);
        for y in 0..self.rows.len() {
            for x in 0..self.rows[y].len() {
                let n = &self.rows[y][x];
                if n.edges.is_empty() || n.room.is_some() {
                    continue;
                }
                if let Some(i) = self.next_room_index(x as i32, y as i32, &rooms) {
                    let r = rooms.remove(i);
                    self.rows[y][x].room = Some(r);
                }
            }
        }
        for n in self.rows.iter_mut().flatten() {
            if !n.edges.is_empty() && n.room.is_none() {
                n.room = Some(RoomType::Monster);
            }
        }
    }

    /// `getNextRoomTypeAccordingToRules`.
    fn next_room_index(&self, x: i32, y: i32, rooms: &[RoomType]) -> Option<usize> {
        let parents = &self.node(x, y).parents;
        let mut siblings = Vec::new();
        for &(px, py) in parents {
            for &(ex, ey) in &self.node(px, py).edges {
                if (ex, ey) != (x, y) {
                    siblings.push((ex, ey));
                }
            }
        }
        for (i, &room) in rooms.iter().enumerate() {
            let assignable = if y <= 4 && matches!(room, RoomType::Rest | RoomType::Elite) {
                false
            } else {
                y < 13 || room != RoomType::Rest
            };
            if !assignable {
                continue;
            }
            let parent_match = matches!(room, RoomType::Rest | RoomType::Treasure | RoomType::Shop | RoomType::Elite)
                && parents.iter().any(|&(px, py)| self.node(px, py).room == Some(room));
            let sibling_match = matches!(room, RoomType::Rest | RoomType::Monster | RoomType::Event | RoomType::Elite | RoomType::Shop)
                && siblings.iter().any(|&(sx, sy)| self.node(sx, sy).room == Some(room));
            if !parent_match && !sibling_match {
                return Some(i);
            }
            if y == 0 {
                return Some(i);
            }
        }
        None
    }

    /// Text rendering like `MapGenerator.toString(nodes, true)`.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        for y in (0..self.rows.len()).rev() {
            s.push_str("\n      ");
            for n in &self.rows[y] {
                let (mut l, mut m, mut r) = (' ', ' ', ' ');
                for &(ex, _) in &n.edges {
                    if ex < n.x {
                        l = '\\';
                    }
                    if ex == n.x {
                        m = '|';
                    }
                    if ex > n.x {
                        r = '/';
                    }
                }
                s.push(l);
                s.push(m);
                s.push(r);
            }
            s.push_str(&format!("\n{:<6}", y));
            for n in &self.rows[y] {
                let sym = if y != self.rows.len() - 1 {
                    if !n.edges.is_empty() {
                        n.room.map_or('*', |r| r.symbol())
                    } else {
                        ' '
                    }
                } else if self.rows[y - 1].iter().any(|ln| ln.edges.iter().any(|e| e.0 == n.x)) {
                    n.room.map_or('*', |r| r.symbol())
                } else {
                    ' '
                };
                s.push(' ');
                s.push(sym);
                s.push(' ');
            }
        }
        s
    }
}
