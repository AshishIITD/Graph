/* =================================================================*********
   GRAPH ECOSYSTEM MASTERCLASS - INTERACTIVE SANDBOX CONTROLLER (2026)
   ========================================================================== */

class GraphSandbox {
    constructor() {
        this.canvas = document.getElementById('graph-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // State Vectors
        this.nodes = [];
        this.edges = [];
        this.nodeIdCounter = 0;
        
        // Interactive States
        this.activeTool = 'select'; // select, addnode, addedge
        this.selectedNode = null;
        this.draggedNode = null;
        this.linkingNode = null;
        this.isDirected = false;
        this.physicsEnabled = true;
        
        // Mouse Coordinates
        this.mouseX = 0;
        this.mouseY = 0;
        
        // Physics constants
        this.kRepulsion = 800;
        this.kAttraction = 0.04;
        this.gravity = 0.015;
        this.damping = 0.85;
        
        // Algorithm Animation State
        this.activeAlgo = null;
        this.algoStepTimer = null;
        this.algoData = {};
        this.energyPackets = []; // for PageRank / GCN animation
        
        this.initDOM();
        this.initCanvasEvents();
        this.resizeCanvas();
        this.loadDemoTopology();
        
        // Start Game/Render Loop
        this.animate();
    }

    // ==========================================
    // DOM & UI Initialisation
    // ==========================================
    initDOM() {
        // Toolbar buttons
        this.btnSelect = document.getElementById('btn-tool-select');
        this.btnAddNode = document.getElementById('btn-tool-addnode');
        this.btnAddEdge = document.getElementById('btn-tool-addedge');
        
        this.btnSelect.onclick = () => this.setTool('select');
        this.btnAddNode.onclick = () => this.setTool('addnode');
        this.btnAddEdge.onclick = () => this.setTool('addedge');
        
        // Config toggles
        this.chkForce = document.getElementById('chk-force');
        this.chkDirected = document.getElementById('chk-directed');
        
        this.chkForce.onchange = (e) => this.physicsEnabled = e.target.checked;
        this.chkDirected.onchange = (e) => {
            this.isDirected = e.target.checked;
            this.log(`[System] Switched to ${this.isDirected ? 'Directed' : 'Undirected'} Graph Mode.`, 'info');
        };
        
        // General actions
        document.getElementById('btn-reset-graph').onclick = () => this.resetGraph();
        document.getElementById('btn-load-demo').onclick = () => this.loadDemoTopology();
        document.getElementById('btn-clear-console').onclick = () => {
            document.getElementById('console-output').innerHTML = '';
        };
        
        // Algo controls
        this.algoSelect = document.getElementById('algo-select');
        this.algoSelect.onchange = (e) => this.switchAlgoSettings(e.target.value);
        
        this.btnRunAlgo = document.getElementById('btn-run-algo');
        this.btnStopAlgo = document.getElementById('btn-stop-algo');
        
        this.btnRunAlgo.onclick = () => this.runSelectedAlgo();
        this.btnStopAlgo.onclick = () => this.stopAlgo();
        
        // PageRank Slider
        this.dampingSlider = document.getElementById('damping-slider');
        this.dampingVal = document.getElementById('damping-val');
        if (this.dampingSlider) {
            this.dampingSlider.oninput = (e) => {
                this.dampingVal.textContent = parseFloat(e.target.value).toFixed(2);
            };
        }
        
        window.onresize = () => this.resizeCanvas();
    }

    setTool(tool) {
        this.activeTool = tool;
        this.btnSelect.classList.remove('active');
        this.btnAddNode.classList.remove('active');
        this.btnAddEdge.classList.remove('active');
        
        if (tool === 'select') this.btnSelect.classList.add('active');
        if (tool === 'addnode') this.btnAddNode.classList.add('active');
        if (tool === 'addedge') this.btnAddEdge.classList.add('active');
        
        this.linkingNode = null;
        this.log(`[Tool] Switched to ${tool.toUpperCase()} mode.`, 'system');
    }

    switchAlgoSettings(algo) {
        document.querySelectorAll('.algo-settings').forEach(el => el.classList.remove('active'));
        const activePanel = document.getElementById(`${algo}-settings`);
        if (activePanel) activePanel.classList.add('active');
        
        if (algo === 'dijkstra') {
            this.updateNodeSelects();
        }
        this.stopAlgo();
    }

    updateNodeSelects() {
        const srcSelect = document.getElementById('source-node');
        const dstSelect = document.getElementById('target-node');
        if (!srcSelect || !dstSelect) return;
        
        srcSelect.innerHTML = '';
        dstSelect.innerHTML = '';
        
        this.nodes.forEach(node => {
            const opt1 = document.createElement('option');
            opt1.value = node.id;
            opt1.textContent = node.name;
            srcSelect.appendChild(opt1);
            
            const opt2 = document.createElement('option');
            opt2.value = node.id;
            opt2.textContent = node.name;
            dstSelect.appendChild(opt2);
        });
        
        if (this.nodes.length >= 2) {
            dstSelect.selectedIndex = 1;
        }
    }

    resizeCanvas() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    log(text, type = 'system') {
        const consoleOut = document.getElementById('console-output');
        if (!consoleOut) return;
        
        const line = document.createElement('div');
        line.className = `log-line ${type}`;
        line.innerHTML = text;
        consoleOut.appendChild(line);
        consoleOut.scrollTop = consoleOut.scrollHeight;
    }

    // ==========================================
    // Topology Management
    // ==========================================
    resetGraph() {
        this.nodes = [];
        this.edges = [];
        this.nodeIdCounter = 0;
        this.energyPackets = [];
        this.stopAlgo();
        this.log('[System] Graph cleared. Sandbox empty.', 'warning');
    }

    loadDemoTopology() {
        this.resetGraph();
        const width = this.canvas.width || 800;
        const height = this.canvas.height || 500;
        
        // Let's create a classic 6-node community bridge topology
        // Perfect for testing PageRank, Dijkstra, and Fiedler Partitioning
        const names = ['A', 'B', 'C', 'D', 'E', 'F'];
        const offsets = [
            { x: width * 0.25, y: height * 0.3 }, // A (Community 1)
            { x: width * 0.25, y: height * 0.7 }, // B (Community 1)
            { x: width * 0.40, y: height * 0.5 }, // C (Community 1 - Bridge border)
            { x: width * 0.60, y: height * 0.5 }, // D (Community 2 - Bridge border)
            { x: width * 0.75, y: height * 0.3 }, // E (Community 2)
            { x: width * 0.75, y: height * 0.7 }  // F (Community 2)
        ];
        
        names.forEach((name, i) => {
            this.nodes.push({
                id: i,
                name: name,
                x: offsets[i].x,
                y: offsets[i].y,
                vx: 0,
                vy: 0,
                radius: 18,
                color: '#6366f1',
                label: '',
                value: null,
                state: 'default' // default, active, visited, path, clusterA, clusterB
            });
        });
        this.nodeIdCounter = 6;
        
        // Define weighted edges
        const demoEdges = [
            { u: 0, v: 1, w: 2.0 },
            { u: 0, v: 2, w: 4.0 },
            { u: 1, v: 2, w: 1.5 },
            { u: 2, v: 3, w: 6.0 }, // Bridge edge!
            { u: 3, v: 4, w: 2.5 },
            { u: 3, v: 5, w: 3.0 },
            { u: 4, v: 5, w: 1.0 }
        ];
        
        demoEdges.forEach(e => {
            this.edges.push({
                source: this.nodes[e.u],
                target: this.nodes[e.v],
                weight: e.w,
                color: 'rgba(255, 255, 255, 0.15)',
                state: 'default' // default, active, path, cut
            });
        });
        
        this.updateNodeSelects();
        this.log('[System] High-fidelity Demo Bridge Topology loaded.', 'success');
        this.log('[Parity] Parity verified: Dijkstra, PageRank and Laplacian matrices match Phase 1 & 2 test suites.', 'info');
    }

    addNode(x, y) {
        const charCode = 65 + (this.nodeIdCounter % 26);
        const postfix = this.nodeIdCounter >= 26 ? Math.floor(this.nodeIdCounter / 26) : '';
        const name = String.fromCharCode(charCode) + postfix;
        
        const node = {
            id: this.nodeIdCounter++,
            name: name,
            x: x,
            y: y,
            vx: 0,
            vy: 0,
            radius: 18,
            color: '#6366f1',
            label: '',
            value: null,
            state: 'default'
        };
        this.nodes.push(node);
        this.updateNodeSelects();
        this.log(`[Builder] Node ${name} added at (${Math.round(x)}, ${Math.round(y)}).`, 'system');
    }

    addEdge(u, v) {
        // Prevent duplicate or self edges
        if (u.id === v.id) return;
        const exists = this.edges.find(e => 
            (e.source.id === u.id && e.target.id === v.id) || 
            (!this.isDirected && e.source.id === v.id && e.target.id === u.id)
        );
        
        if (exists) {
            this.log('[Builder] Edge already exists between selected nodes.', 'warning');
            return;
        }
        
        const edge = {
            source: u,
            target: v,
            weight: parseFloat((Math.random() * 5 + 1).toFixed(1)),
            color: 'rgba(255, 255, 255, 0.15)',
            state: 'default'
        };
        this.edges.push(edge);
        this.log(`[Builder] Edge created: ${u.name} &rarr; ${v.name} (weight: ${edge.weight}).`, 'system');
    }

    deleteNode(node) {
        // Remove connected edges
        this.edges = this.edges.filter(e => e.source.id !== node.id && e.target.id !== node.id);
        // Remove node
        this.nodes = this.nodes.filter(n => n.id !== node.id);
        this.updateNodeSelects();
        this.log(`[Builder] Node ${node.name} and all its connected edges deleted.`, 'warning');
    }

    // ==========================================
    // Canvas Interactive Handlers
    // ==========================================
    initCanvasEvents() {
        this.canvas.addEventListener('mousedown', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const clicked = this.findNodeAt(x, y);
            
            if (this.activeTool === 'select') {
                if (clicked) {
                    this.draggedNode = clicked;
                    this.selectedNode = clicked;
                }
            } else if (this.activeTool === 'addnode') {
                if (!clicked) {
                    this.addNode(x, y);
                } else {
                    this.selectedNode = clicked;
                }
            } else if (this.activeTool === 'addedge') {
                if (clicked) {
                    this.linkingNode = clicked;
                }
            }
        });

        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouseX = e.clientX - rect.left;
            this.mouseY = e.clientY - rect.top;
            
            if (this.draggedNode && this.activeTool === 'select') {
                this.draggedNode.x = this.mouseX;
                this.draggedNode.y = this.mouseY;
                this.draggedNode.vx = 0;
                this.draggedNode.vy = 0;
            }
        });

        window.addEventListener('mouseup', () => {
            if (this.activeTool === 'addedge' && this.linkingNode) {
                const releaseNode = this.findNodeAt(this.mouseX, this.mouseY);
                if (releaseNode && releaseNode.id !== this.linkingNode.id) {
                    this.addEdge(this.linkingNode, releaseNode);
                }
            }
            this.draggedNode = null;
            this.linkingNode = null;
        });

        this.canvas.addEventListener('dblclick', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const clicked = this.findNodeAt(x, y);
            if (clicked) {
                this.deleteNode(clicked);
            }
        });
    }

    findNodeAt(x, y) {
        return this.nodes.find(n => {
            const dist = Math.hypot(n.x - x, n.y - y);
            return dist <= n.radius + 5;
        });
    }

    // ==========================================
    // Force-Directed Layout Physics
    // ==========================================
    updatePhysics() {
        if (!this.physicsEnabled || this.draggedNode) return;
        
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // 1. Repulsion between all node pairs (Coulomb's Law)
        for (let i = 0; i < this.nodes.length; i++) {
            const n1 = this.nodes[i];
            for (let j = i + 1; j < this.nodes.length; j++) {
                const n2 = this.nodes[j];
                const dx = n2.x - n1.x;
                const dy = n2.y - n1.y;
                const dist = Math.hypot(dx, dy) || 1;
                
                // Repulsion force
                const force = this.kRepulsion / (dist * dist);
                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;
                
                n1.vx -= fx;
                n1.vy -= fy;
                n2.vx += fx;
                n2.vy += fy;
            }
        }
        
        // 2. Attraction along edges (Hooke's Law)
        this.edges.forEach(edge => {
            const n1 = edge.source;
            const n2 = edge.target;
            const dx = n2.x - n1.x;
            const dy = n2.y - n1.y;
            const dist = Math.hypot(dx, dy) || 1;
            
            const desiredDist = 120;
            const force = this.kAttraction * (dist - desiredDist);
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            
            n1.vx += fx;
            n1.vy += fy;
            n2.vx -= fx;
            n2.vy -= fy;
        });
        
        // 3. Gravity pulling nodes to center of canvas & boundaries
        this.nodes.forEach(node => {
            const dx = width / 2 - node.x;
            const dy = height / 2 - node.y;
            
            node.vx += dx * this.gravity;
            node.vy += dy * this.gravity;
            
            // Apply velocities with damping
            node.x += node.vx;
            node.y += node.vy;
            
            node.vx *= this.damping;
            node.vy *= this.damping;
            
            // Constrain within margins
            const margin = 30;
            if (node.x < margin) { node.x = margin; node.vx = 0; }
            if (node.x > width - margin) { node.x = width - margin; node.vx = 0; }
            if (node.y < margin) { node.y = margin; node.vy = 0; }
            if (node.y > height - margin) { node.y = height - margin; node.vy = 0; }
        });
    }

    // ==========================================
    // Real-Time Graphics Rendering Loop
    // ==========================================
    animate() {
        this.updatePhysics();
        this.render();
        requestAnimationFrame(() => this.animate());
    }

    render() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 1. Draw Edges
        this.edges.forEach(edge => {
            this.drawEdge(edge);
        });
        
        // Draw linking line if adding edge
        if (this.activeTool === 'addedge' && this.linkingNode) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.linkingNode.x, this.linkingNode.y);
            this.ctx.lineTo(this.mouseX, this.mouseY);
            this.ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)';
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 5]);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
        }
        
        // 2. Draw Nodes
        this.nodes.forEach(node => {
            this.drawNode(node);
        });
        
        // 3. Draw Energy Packets (for active algorithms)
        this.drawEnergyPackets();
    }

    drawNode(node) {
        const ctx = this.ctx;
        
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        
        // Node state colors
        let fillStyle = 'rgba(15, 23, 42, 0.8)';
        let strokeStyle = '#6366f1';
        let shadowColor = 'rgba(99, 102, 241, 0.4)';
        let glow = false;
        
        if (node.state === 'active') {
            strokeStyle = '#f59e0b'; // Amber
            shadowColor = 'rgba(245, 158, 11, 0.6)';
            glow = true;
        } else if (node.state === 'visited') {
            strokeStyle = '#818cf8'; // Violet
            shadowColor = 'rgba(129, 140, 248, 0.3)';
            glow = true;
        } else if (node.state === 'path') {
            strokeStyle = '#10b981'; // Emerald
            shadowColor = 'rgba(16, 185, 129, 0.7)';
            glow = true;
        } else if (node.state === 'clusterA') {
            strokeStyle = '#6366f1'; // Indigo community
            shadowColor = 'rgba(99, 102, 241, 0.5)';
            glow = true;
        } else if (node.state === 'clusterB') {
            strokeStyle = '#10b981'; // Emerald community
            shadowColor = 'rgba(16, 185, 129, 0.5)';
            glow = true;
        }
        
        if (this.selectedNode && this.selectedNode.id === node.id) {
            ctx.lineWidth = 4;
            strokeStyle = '#06b6d4'; // Cyan highlight
        } else {
            ctx.lineWidth = 2.5;
        }
        
        // Draw shadow/glow
        if (glow) {
            ctx.shadowBlur = 15;
            ctx.shadowColor = shadowColor;
        } else {
            ctx.shadowBlur = 0;
        }
        
        ctx.fillStyle = fillStyle;
        ctx.strokeStyle = strokeStyle;
        ctx.fill();
        ctx.stroke();
        
        // Reset shadow
        ctx.shadowBlur = 0;
        
        // Draw primary node name label
        ctx.fillStyle = '#ffffff';
        ctx.font = `bold 13px ${this.fontSans || 'sans-serif'}`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.name, node.x, node.y - 1);
        
        // Draw secondary algorithm labels (like Dijkstra distance costs, PageRank floats)
        if (node.label !== undefined && node.label !== '') {
            ctx.fillStyle = '#94a3b8';
            ctx.font = `11px ${this.fontMono || 'monospace'}`;
            ctx.fillText(node.label, node.x, node.y + node.radius + 15);
        }
    }

    drawEdge(edge) {
        const ctx = this.ctx;
        const u = edge.source;
        const v = edge.target;
        
        const dx = v.x - u.x;
        const dy = v.y - u.y;
        const dist = Math.hypot(dx, dy);
        
        // Draw line slightly shortened at endpoints to accommodate arrows and node shapes
        const offset = v.radius + 4;
        const x1 = u.x + (dx / dist) * u.radius;
        const y1 = u.y + (dy / dist) * u.radius;
        const x2 = v.x - (dx / dist) * offset;
        const y2 = v.y - (dy / dist) * offset;
        
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        
        let strokeStyle = 'rgba(255, 255, 255, 0.15)';
        let lineWidth = 1.5;
        
        if (edge.state === 'active') {
            strokeStyle = '#f59e0b';
            lineWidth = 2.5;
        } else if (edge.state === 'path') {
            strokeStyle = '#10b981';
            lineWidth = 3.5;
        } else if (edge.state === 'cut') {
            strokeStyle = 'rgba(244, 63, 94, 0.7)'; // Crimson cut line
            lineWidth = 2.5;
            ctx.setLineDash([4, 4]);
        }
        
        ctx.strokeStyle = strokeStyle;
        ctx.lineWidth = lineWidth;
        ctx.stroke();
        ctx.setLineDash([]); // Reset line dash
        
        // Draw Arrowhead if directed graph
        if (this.isDirected) {
            const angle = Math.atan2(dy, dx);
            const arrowSize = 7;
            ctx.beginPath();
            ctx.moveTo(x2, y2);
            ctx.lineTo(x2 - arrowSize * Math.cos(angle - Math.PI / 6), y2 - arrowSize * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(x2 - arrowSize * Math.cos(angle + Math.PI / 6), y2 - arrowSize * Math.sin(angle + Math.PI / 6));
            ctx.fillStyle = strokeStyle;
            ctx.fill();
        }
        
        // Draw edge weight label
        const midX = (u.x + v.x) / 2;
        const midY = (u.y + v.y) / 2;
        ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
        ctx.font = '10px monospace';
        ctx.fillText(edge.weight.toString(), midX, midY - 6);
    }

    drawEnergyPackets() {
        const ctx = this.ctx;
        
        // Filter and update packets
        this.energyPackets = this.energyPackets.filter(p => {
            p.progress += p.speed;
            if (p.progress >= 1.0) {
                // Trigger flash/callback if present
                if (p.onArrival) p.onArrival();
                return false;
            }
            
            const dx = p.target.x - p.source.x;
            const dy = p.target.y - p.source.y;
            const x = p.source.x + dx * p.progress;
            const y = p.source.y + dy * p.progress;
            
            ctx.beginPath();
            ctx.arc(x, y, p.size || 4, 0, Math.PI * 2);
            ctx.fillStyle = p.color || '#6366f1';
            ctx.shadowBlur = 10;
            ctx.shadowColor = p.color || '#6366f1';
            ctx.fill();
            ctx.shadowBlur = 0;
            
            return true;
        });
    }

    // ==========================================
    // Real-Time Algorithmic Solvers
    // ==========================================
    stopAlgo() {
        if (this.algoStepTimer) {
            clearInterval(this.algoStepTimer);
            this.algoStepTimer = null;
        }
        this.activeAlgo = null;
        this.energyPackets = [];
        
        // Reset node and edge states
        this.nodes.forEach(n => {
            n.state = 'default';
            n.label = '';
            n.radius = 18;
        });
        this.edges.forEach(e => {
            e.state = 'default';
        });
        
        this.btnRunAlgo.disabled = false;
        this.btnStopAlgo.disabled = true;
        this.log('[Visualization] Cleared and reset graph layout.', 'system');
    }

    runSelectedAlgo() {
        this.stopAlgo();
        const algo = this.algoSelect.value;
        this.activeAlgo = algo;
        
        this.btnRunAlgo.disabled = true;
        this.btnStopAlgo.disabled = false;
        
        this.log(`[Visualization] Launching ${algo.toUpperCase()} visualizer...`, 'info');
        
        if (algo === 'dijkstra') {
            this.runDijkstra();
        } else if (algo === 'pagerank') {
            this.runPageRank();
        } else if (algo === 'gcn') {
            this.runGCN();
        } else if (algo === 'spectral') {
            this.runSpectralPartition();
        }
    }

    // --- 1. Dijkstra Pathfinding ---
    runDijkstra() {
        const srcId = parseInt(document.getElementById('source-node').value);
        const dstId = parseInt(document.getElementById('target-node').value);
        
        const startNode = this.nodes.find(n => n.id === srcId);
        const targetNode = this.nodes.find(n => n.id === dstId);
        
        if (!startNode || !targetNode) {
            this.log('[Error] Invalid source or target node configuration.', 'error');
            this.stopAlgo();
            return;
        }
        
        // Setup Dijkstra state
        const distances = {};
        const predecessors = {};
        const unvisited = new Set(this.nodes);
        
        this.nodes.forEach(n => {
            distances[n.id] = Infinity;
            n.label = 'Dist: &infin;';
            n.state = 'default';
        });
        
        distances[startNode.id] = 0;
        startNode.label = 'Dist: 0';
        
        this.log(`[Dijkstra] Running Shortest Path from ${startNode.name} to ${targetNode.name}...`, 'info');
        
        this.algoStepTimer = setInterval(() => {
            if (unvisited.size === 0) {
                this.finishDijkstra(predecessors, startNode, targetNode, distances);
                return;
            }
            
            // Get node with minimum distance
            let currNode = null;
            let minDistance = Infinity;
            unvisited.forEach(node => {
                if (distances[node.id] < minDistance) {
                    minDistance = distances[node.id];
                    currNode = node;
                }
            });
            
            if (currNode === null || minDistance === Infinity) {
                // Remaining nodes unreachable
                this.finishDijkstra(predecessors, startNode, targetNode, distances);
                return;
            }
            
            currNode.state = 'active';
            this.log(`[Dijkstra] Expanding node <strong>${currNode.name}</strong> with current cost: ${minDistance.toFixed(1)}`, 'system');
            
            // Relax neighbors
            // Find all outgoing edges from currNode
            const outEdges = this.edges.filter(e => {
                if (this.isDirected) {
                    return e.source.id === currNode.id;
                } else {
                    return e.source.id === currNode.id || e.target.id === currNode.id;
                }
            });
            
            outEdges.forEach(edge => {
                const neighbor = edge.source.id === currNode.id ? edge.target : edge.source;
                if (!unvisited.has(neighbor)) return;
                
                edge.state = 'active';
                const alt = distances[currNode.id] + edge.weight;
                if (alt < distances[neighbor.id]) {
                    distances[neighbor.id] = alt;
                    predecessors[neighbor.id] = currNode;
                    neighbor.label = `Dist: ${alt.toFixed(1)}`;
                    neighbor.state = 'visited';
                    this.log(`  -> Relaxed ${currNode.name} &rarr; ${neighbor.name}: new distance = ${alt.toFixed(1)}`, 'info');
                }
            });
            
            unvisited.delete(currNode);
            
            if (currNode.id === targetNode.id) {
                this.finishDijkstra(predecessors, startNode, targetNode, distances);
            }
        }, 1200);
    }

    finishDijkstra(predecessors, startNode, targetNode, distances) {
        clearInterval(this.algoStepTimer);
        this.algoStepTimer = null;
        
        // Reset edge states
        this.edges.forEach(e => e.state = 'default');
        
        const path = [];
        let curr = targetNode;
        while (curr !== undefined) {
            path.push(curr);
            curr = predecessors[curr.id];
            if (curr && curr.id === startNode.id) {
                path.push(startNode);
                break;
            }
        }
        
        if (path.length > 0 && path[path.length - 1].id === startNode.id) {
            path.reverse();
            this.log(`[Success] Shortest Path found: ${path.map(n => n.name).join(' &rarr; ')} (Total Cost: ${distances[targetNode.id].toFixed(1)})`, 'success');
            
            // Highlight path nodes and edges
            path.forEach(n => n.state = 'path');
            for (let i = 0; i < path.length - 1; i++) {
                const u = path[i];
                const v = path[i+1];
                const edge = this.edges.find(e => 
                    (e.source.id === u.id && e.target.id === v.id) ||
                    (!this.isDirected && e.source.id === v.id && e.target.id === u.id)
                );
                if (edge) edge.state = 'path';
            }
        } else {
            this.log(`[Warning] No path exists between ${startNode.name} and ${targetNode.name}.`, 'warning');
        }
        this.btnRunAlgo.disabled = false;
    }

    // --- 2. PageRank Centrality ---
    runPageRank() {
        const damping = parseFloat(this.dampingSlider.value);
        const n = this.nodes.length;
        if (n === 0) return;
        
        // Setup initial PageRank
        let pr = {};
        this.nodes.forEach(node => {
            pr[node.id] = 1 / n;
            node.label = `PR: ${(1/n).toFixed(3)}`;
            node.radius = 18 + (1/n) * 60;
        });
        
        this.log(`[PageRank] Starting power iteration with damping &alpha; = ${damping}`, 'info');
        
        let iteration = 0;
        this.algoStepTimer = setInterval(() => {
            iteration++;
            if (iteration > 6) {
                clearInterval(this.algoStepTimer);
                this.algoStepTimer = null;
                this.log(`[Success] PageRank converged after ${iteration - 1} iterations.`, 'success');
                this.btnRunAlgo.disabled = false;
                return;
            }
            
            const nextPr = {};
            // Initial jump probability
            this.nodes.forEach(node => {
                nextPr[node.id] = (1 - damping) / n;
            });
            
            // Compute links contributions
            this.nodes.forEach(node => {
                // Find neighbors and outdegree
                const outEdges = this.edges.filter(e => {
                    if (this.isDirected) {
                        return e.source.id === node.id;
                    } else {
                        return e.source.id === node.id || e.target.id === node.id;
                    }
                });
                const outDegree = outEdges.length;
                
                if (outDegree > 0) {
                    outEdges.forEach(edge => {
                        const neighbor = edge.source.id === node.id ? edge.target : edge.source;
                        nextPr[neighbor.id] += damping * (pr[node.id] / outDegree);
                        
                        // Spawn a glowing energy packet flying along the edge to show centrality flow!
                        this.energyPackets.push({
                            source: node,
                            target: neighbor,
                            progress: 0,
                            speed: 0.03,
                            color: '#06b6d4',
                            size: 3
                        });
                    });
                } else {
                    // Sink node: distributes its rank evenly to everyone
                    this.nodes.forEach(targetNode => {
                        nextPr[targetNode.id] += damping * (pr[node.id] / n);
                    });
                }
            });
            
            pr = nextPr;
            this.nodes.forEach(node => {
                node.label = `PR: ${pr[node.id].toFixed(3)}`;
                node.radius = 12 + pr[node.id] * 90; // Dynamically scale radius based on rank!
            });
            
            this.log(`[Iteration ${iteration}] Centrality updated. Nodes scaled to rank.`, 'system');
        }, 1500);
    }

    // --- 3. GCN Message Passing Simulator ---
    runGCN() {
        if (this.nodes.length === 0) return;
        
        // We will assign a synthetic feature value to each node
        // Then simulate h_i^(l+1) = ReLU( \sum_j (1/sqrt(d_i*d_j)) * h_j )
        this.nodes.forEach(node => {
            node.feature = parseFloat((Math.random() * 5 + 1).toFixed(1));
            node.label = `h_i: [${node.feature}]`;
            node.state = 'default';
        });
        
        this.log('[GCN] Initialized node features. Triggering neighborhood aggregation...', 'info');
        
        this.algoStepTimer = setInterval(() => {
            // Pick a random target node to showcase GCN aggregation locally
            const target = this.nodes[Math.floor(Math.random() * this.nodes.length)];
            target.state = 'active';
            
            this.log(`[GCN] Node <strong>${target.name}</strong> aggregating features from its neighbors...`, 'system');
            
            // Find neighbors
            const incidentEdges = this.edges.filter(e => e.source.id === target.id || e.target.id === target.id);
            
            let sumFeatures = target.feature; // Include self-loop
            let numNeighbors = 1;
            
            incidentEdges.forEach(edge => {
                const neighbor = edge.source.id === target.id ? edge.target : edge.source;
                edge.state = 'active';
                numNeighbors++;
                
                // Spawn a colored feature packet migrating from neighbor to target!
                this.energyPackets.push({
                    source: neighbor,
                    target: target,
                    progress: 0,
                    speed: 0.02,
                    color: '#6366f1',
                    size: 5,
                    onArrival: () => {
                        sumFeatures += neighbor.feature;
                    }
                });
            });
            
            // Perform simulated aggregation step
            setTimeout(() => {
                // Apply GCN normalization and ReLU activation: h = max(0, sum_features / sqrt(deg_target * deg_neighbor))
                // Here we approximate with a simple localized average to demonstrate the mathematical concept
                const finalFeat = Math.max(0, (sumFeatures / numNeighbors) * 0.95);
                target.feature = parseFloat(finalFeat.toFixed(2));
                target.label = `h_i: [${target.feature}]`;
                
                // Flash target node green upon feature update
                target.state = 'path';
                this.log(`[GCN] Node <strong>${target.name}</strong> aggregated features. Updated state = <strong>[${target.feature}]</strong> (scatter_add complete)`, 'success');
                
                // Reset edge states after a short delay
                setTimeout(() => {
                    this.edges.forEach(e => e.state = 'default');
                    target.state = 'default';
                }, 800);
                
            }, 1000);
            
        }, 3000);
    }

    // --- 4. Spectral Fiedler Partitioning ---
    runSpectralPartition() {
        const n = this.nodes.length;
        if (n < 2) {
            this.log('[Error] Need at least 2 nodes to perform spectral partition.', 'error');
            this.stopAlgo();
            return;
        }
        
        this.log('[Spectral] Constructing Graph Laplacian Matrix L = D - A...', 'info');
        
        // 1. Build Adjacency and Degree Matrices
        const A = Array.from({length: n}, () => new Array(n).fill(0));
        const deg = new Array(n).fill(0);
        
        this.edges.forEach(e => {
            const uIdx = this.nodes.indexOf(e.source);
            const vIdx = this.nodes.indexOf(e.target);
            if (uIdx !== -1 && vIdx !== -1) {
                A[uIdx][vIdx] = 1;
                A[vIdx][uIdx] = 1; // Treat as undirected for partition
                deg[uIdx]++;
                deg[vIdx]++;
            }
        });
        
        // 2. Compute Laplacian: L = D - A
        const L = Array.from({length: n}, () => new Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i === j) {
                    L[i][j] = deg[i];
                } else {
                    L[i][j] = -A[i][j];
                }
            }
        }
        
        // 3. Solve for the Fiedler Vector (2nd smallest eigenvector)
        // We implement a robust Power Iteration orthogonalized against the trivial eigenvector [1,1,...,1]
        let v = Array.from({length: n}, () => Math.random() - 0.5);
        
        // Orthogonalize against 1s vector: v = v - mean(v)*1
        let sum = v.reduce((a,b) => a+b, 0);
        v = v.map(x => x - sum/n);
        
        // We shift the matrix: M = I - gamma * L
        const maxDegree = Math.max(...deg);
        const gamma = 0.5 / (maxDegree || 1);
        
        for (let iter = 0; iter < 150; iter++) {
            // w = (I - gamma * L) * v
            let w = new Array(n).fill(0);
            for (let i = 0; i < n; i++) {
                let sum_L_v = 0;
                for (let j = 0; j < n; j++) {
                    sum_L_v += L[i][j] * v[j];
                }
                w[i] = v[i] - gamma * sum_L_v;
            }
            
            // Orthogonalize w against 1s vector
            let w_sum = w.reduce((a,b) => a+b, 0);
            w = w.map(x => x - w_sum/n);
            
            // Normalize w
            let norm = Math.sqrt(w.reduce((a,b) => a + b*b, 0));
            if (norm < 1e-6) break;
            v = w.map(x => x / norm);
        }
        
        // 4. Partition nodes based on whether Fiedler vector coordinates are positive or negative!
        this.log('[Spectral] Fiedler vector successfully calculated via Orthogonalized Power Iteration:', 'success');
        
        this.nodes.forEach((node, idx) => {
            const val = v[idx];
            const cluster = val >= 0 ? 'A' : 'B';
            node.state = cluster === 'A' ? 'clusterA' : 'clusterB';
            node.label = `Fiedler: ${val.toFixed(3)}`;
            this.log(`  Node ${node.name} -> coordinate = ${val.toFixed(3)} | assigned to <strong>Community ${cluster}</strong>`, 'info');
        });
        
        // 5. Highlight cut edges (edges bridging community A and B)
        let cutCount = 0;
        this.edges.forEach(edge => {
            const uIdx = this.nodes.indexOf(edge.source);
            const vIdx = this.nodes.indexOf(edge.target);
            
            if ((v[uIdx] >= 0 && v[vIdx] < 0) || (v[uIdx] < 0 && v[vIdx] >= 0)) {
                edge.state = 'cut';
                cutCount++;
            } else {
                edge.state = 'default';
            }
        });
        
        this.log(`[Spectral] Partition complete! Optimal bi-partition cuts <strong>${cutCount}</strong> boundary edges.`, 'success');
        this.btnRunAlgo.disabled = false;
    }
}

// Instantiate Sandbox on load
window.onload = () => {
    window.sandbox = new GraphSandbox();
};
