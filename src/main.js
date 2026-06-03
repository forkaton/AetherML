// =============================================
// Main.js — Module Router & Global Coordinator
// =============================================

document.addEventListener('DOMContentLoaded', () => {
    const navButtons = document.querySelectorAll('.nav-btn');
    const modules = document.querySelectorAll('.module-view');
    const mathSections = document.querySelectorAll('.math-module-content');

    // Map module IDs to their math panel and init functions
    const moduleMap = {
        'module-regression': { math: 'reg-math-content', init: null }, // regression inits itself
        'module-knn':        { math: 'knn-math-content', init: () => window.AetherML?.knn?.init() },
        'module-kmeans':     { math: 'kmeans-math-content', init: () => window.AetherML?.kmeans?.init() },
        'module-tree':       { math: 'tree-math-content', init: () => window.AetherML?.tree?.init() },
        'module-nn':         { math: 'nn-math-content', init: () => window.AetherML?.nn?.init() }
    };

    function switchModule(targetId) {
        // Update nav buttons
        navButtons.forEach(b => b.classList.remove('active'));
        const activeBtn = document.querySelector(`[data-target="${targetId}"]`);
        if (activeBtn) activeBtn.classList.add('active');

        // Show target module, hide others
        modules.forEach(mod => {
            mod.classList.toggle('active', mod.id === targetId);
        });

        // Show corresponding math panel
        mathSections.forEach(sec => sec.classList.remove('active'));
        const mathId = moduleMap[targetId]?.math;
        if (mathId) {
            const mathEl = document.getElementById(mathId);
            if (mathEl) mathEl.classList.add('active');
        }

        // Initialize module if needed
        const initFn = moduleMap[targetId]?.init;
        if (initFn) {
            // Small delay to allow DOM to be visible for canvas sizing
            setTimeout(initFn, 50);
        }
    }

    // Nav click handlers
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            switchModule(btn.getAttribute('data-target'));
        });
    });

    // Math Panel Toggle
    const toggleMathBtn = document.getElementById('toggle-math-panel');
    const mathPanel = document.getElementById('math-panel');
    let panelOpen = true;

    toggleMathBtn.addEventListener('click', () => {
        panelOpen = !panelOpen;
        mathPanel.classList.toggle('collapsed', !panelOpen);
        toggleMathBtn.textContent = panelOpen ? '×' : '◀';
    });
});
