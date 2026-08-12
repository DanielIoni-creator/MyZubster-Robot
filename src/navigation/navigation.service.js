/**
 * 🧭 Navigation Service - Navigazione Autonoma
 */

class NavigationService {
    constructor() {
        this.position = { x: 0, y: 0 };
        this.map = [];
        this.path = [];
        this.obstacles = [];
        this.initMap();
    }

    // Inizializza mappa
    initMap() {
        // Mappa 10x10
        this.map = Array(10).fill(null).map(() => Array(10).fill(0));
        
        // Aggiungi ostacoli
        this.obstacles = [
            { x: 3, y: 3 },
            { x: 4, y: 3 },
            { x: 5, y: 3 },
            { x: 7, y: 5 },
            { x: 7, y: 6 },
        ];
        
        this.obstacles.forEach(obs => {
            if (this.map[obs.x] && this.map[obs.x][obs.y] !== undefined) {
                this.map[obs.x][obs.y] = 1;
            }
        });
    }

    // Calcola percorso
    async findPath(start, end) {
        try {
            // Algoritmo A* semplificato
            const path = this.aStar(start, end);
            this.path = path;
            return {
                success: true,
                path: path,
                steps: path.length,
                message: '✅ Percorso calcolato'
            };
        } catch (error) {
            console.error('❌ Errore pathfinding:', error);
            return { success: false, error: error.message };
        }
    }

    // A* Algorithm
    aStar(start, end) {
        const openSet = [start];
        const closedSet = [];
        const cameFrom = {};
        const gScore = {};
        const fScore = {};

        const startKey = `${start.x},${start.y}`;
        const endKey = `${end.x},${end.y}`;

        gScore[startKey] = 0;
        fScore[startKey] = this.heuristic(start, end);

        while (openSet.length > 0) {
            // Trova il nodo con fScore minimo
            let current = openSet[0];
            let currentIndex = 0;
            openSet.forEach((node, index) => {
                const key = `${node.x},${node.y}`;
                if (fScore[key] < fScore[`${current.x},${current.y}`]) {
                    current = node;
                    currentIndex = index;
                }
            });

            // Se abbiamo raggiunto la fine
            if (current.x === end.x && current.y === end.y) {
                return this.reconstructPath(cameFrom, current);
            }

            // Rimuovi current da openSet
            openSet.splice(currentIndex, 1);
            closedSet.push(current);

            // Controlla i vicini
            const neighbors = this.getNeighbors(current);
            for (const neighbor of neighbors) {
                const neighborKey = `${neighbor.x},${neighbor.y}`;
                
                if (closedSet.some(n => n.x === neighbor.x && n.y === neighbor.y)) {
                    continue;
                }

                // Calcola gScore
                const tentativeGScore = gScore[`${current.x},${current.y}`] + 1;

                if (!openSet.some(n => n.x === neighbor.x && n.y === neighbor.y)) {
                    openSet.push(neighbor);
                } else if (tentativeGScore >= gScore[neighborKey]) {
                    continue;
                }

                cameFrom[neighborKey] = current;
                gScore[neighborKey] = tentativeGScore;
                fScore[neighborKey] = gScore[neighborKey] + this.heuristic(neighbor, end);
            }
        }

        return []; // Nessun percorso trovato
    }

    // Euristica (distanza Manhattan)
    heuristic(a, b) {
        return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
    }

    // Ottieni vicini validi
    getNeighbors(node) {
        const directions = [
            { dx: -1, dy: 0 },
            { dx: 1, dy: 0 },
            { dx: 0, dy: -1 },
            { dx: 0, dy: 1 },
        ];

        const neighbors = [];
        for (const dir of directions) {
            const newX = node.x + dir.dx;
            const newY = node.y + dir.dy;
            
            // Controlla se nella mappa
            if (newX < 0 || newX >= 10 || newY < 0 || newY >= 10) {
                continue;
            }
            
            // Controlla se è un ostacolo
            if (this.map[newX][newY] === 1) {
                continue;
            }
            
            neighbors.push({ x: newX, y: newY });
        }
        
        return neighbors;
    }

    // Ricostruisci percorso
    reconstructPath(cameFrom, current) {
        const path = [current];
        let key = `${current.x},${current.y}`;
        
        while (cameFrom[key]) {
            current = cameFrom[key];
            path.unshift(current);
            key = `${current.x},${current.y}`;
        }
        
        return path;
    }

    // Esegui navigazione
    async navigateTo(target) {
        try {
            const start = { ...this.position };
            const result = await this.findPath(start, target);
            
            if (!result.success) {
                return result;
            }

            // Simula movimento
            for (const step of result.path) {
                this.position = { ...step };
                console.log(`🧭 Movimento a: (${step.x}, ${step.y})`);
                // In produzione, invia comandi al robot
                await new Promise(resolve => setTimeout(resolve, 500));
            }

            return {
                success: true,
                position: this.position,
                path: result.path,
                message: '✅ Navigazione completata'
            };
        } catch (error) {
            console.error('❌ Errore navigazione:', error);
            return { success: false, error: error.message };
        }
    }

    // Evita ostacoli in tempo reale
    async avoidObstacles() {
        try {
            const nearby = [];
            for (const obs of this.obstacles) {
                const dist = Math.abs(obs.x - this.position.x) + Math.abs(obs.y - this.position.y);
                if (dist < 3) {
                    nearby.push({ ...obs, distance: dist });
                }
            }

            if (nearby.length > 0) {
                // Trova la direzione più sicura
                const directions = this.getNeighbors(this.position);
                const safeDirections = directions.filter(d => 
                    !this.obstacles.some(o => o.x === d.x && o.y === d.y)
                );

                if (safeDirections.length > 0) {
                    this.position = safeDirections[0];
                    return {
                        success: true,
                        message: '🔄 Ostacolo evitato',
                        position: this.position
                    };
                }
            }

            return {
                success: true,
                message: '✅ Nessun ostacolo nelle vicinanze'
            };
        } catch (error) {
            console.error('❌ Errore avoidObstacles:', error);
            return { success: false, error: error.message };
        }
    }

    // Ottieni mappa
    async getMap() {
        return {
            success: true,
            map: this.map,
            position: this.position,
            obstacles: this.obstacles
        };
    }
}

export const navigationService = new NavigationService();
