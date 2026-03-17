/**
 * Expert Mode Toggle for Fragile Docs
 *
 * This script provides a toggle switch that allows readers to switch between:
 * - Full Mode: Shows all content including Feynman-style explanatory prose
 * - Expert Mode: Hides explanatory prose, showing only formal mathematical content
 *
 * The preference is persisted in localStorage across sessions.
 */
(function() {
    'use strict';

    const STORAGE_KEY = 'fragile-expert-mode';

    /**
     * Initialize expert mode from saved preference (runs immediately)
     */
    function initExpertMode() {
        const isExpert = localStorage.getItem(STORAGE_KEY) === 'true';
        if (isExpert) {
            document.documentElement.classList.add('expert-mode');
        }
        return isExpert;
    }

    /**
     * Create the toggle switch component
     */
    function createToggleSwitch(isExpert) {
        // Container
        const container = document.createElement('div');
        container.className = 'expert-mode-container';

        // Label "Full"
        const labelFull = document.createElement('span');
        labelFull.className = 'expert-mode-label expert-mode-label-full';
        labelFull.textContent = 'Full';

        // Switch wrapper
        const switchLabel = document.createElement('label');
        switchLabel.className = 'expert-mode-switch';
        switchLabel.setAttribute('title', 'Toggle between Full Mode (with explanations) and Expert Mode (formal content only)');

        // Hidden checkbox
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = isExpert;
        checkbox.setAttribute('aria-label', 'Toggle expert mode');

        // Slider
        const slider = document.createElement('span');
        slider.className = 'expert-mode-slider';

        switchLabel.appendChild(checkbox);
        switchLabel.appendChild(slider);

        // Label "Expert"
        const labelExpert = document.createElement('span');
        labelExpert.className = 'expert-mode-label expert-mode-label-expert';
        labelExpert.textContent = 'Expert';

        // Assemble
        container.appendChild(labelFull);
        container.appendChild(switchLabel);
        container.appendChild(labelExpert);

        // Update active label styling
        function updateLabels(expert) {
            if (expert) {
                labelFull.classList.remove('active');
                labelExpert.classList.add('active');
            } else {
                labelFull.classList.add('active');
                labelExpert.classList.remove('active');
            }
        }
        updateLabels(isExpert);

        // Event handler
        checkbox.addEventListener('change', function() {
            const nowExpert = checkbox.checked;
            document.documentElement.classList.toggle('expert-mode', nowExpert);
            localStorage.setItem(STORAGE_KEY, nowExpert);
            updateLabels(nowExpert);
            announceChange(nowExpert);
        });

        return container;
    }

    /**
     * Announce mode change for accessibility
     */
    function announceChange(isExpert) {
        const announcement = isExpert
            ? 'Expert mode enabled. Explanatory text is now hidden.'
            : 'Full mode enabled. All content is now visible.';

        let liveRegion = document.getElementById('expert-mode-announce');
        if (!liveRegion) {
            liveRegion = document.createElement('div');
            liveRegion.id = 'expert-mode-announce';
            liveRegion.setAttribute('aria-live', 'polite');
            liveRegion.setAttribute('aria-atomic', 'true');
            liveRegion.style.cssText = 'position:absolute;left:-9999px;';
            document.body.appendChild(liveRegion);
        }
        liveRegion.textContent = announcement;
    }

    /**
     * Insert the toggle switch into the page
     */
    function insertToggle(toggle) {
        // Try to find the primary sidebar toggle button
        const primaryToggle = document.querySelector('.sidebar-toggle.primary-toggle');
        if (primaryToggle) {
            const wrapper = primaryToggle.closest('.header-article-item');
            if (wrapper && wrapper.parentNode) {
                // Insert after the sidebar toggle
                wrapper.parentNode.insertBefore(toggle, wrapper.nextSibling);
                return;
            }
        }

        // Fallback: try header article items area
        const headerItems = document.querySelector('.header-article-items');
        if (headerItems) {
            headerItems.appendChild(toggle);
            return;
        }

        // Fallback: fixed position at top-right
        toggle.classList.add('expert-mode-container-fixed');
        document.body.appendChild(toggle);
    }

    /**
     * Apply volume-header class to volume header captions in the sidebar
     * Volume headers are identified by containing "Vol." or "Supplementary" in their text
     */
    function applyVolumeHeaderStyles() {
        // Find all caption elements in the sidebar
        const captions = document.querySelectorAll('.bd-sidebar-primary .caption-text');

        captions.forEach(function(caption) {
            const text = caption.textContent || '';
            // Check if this is a volume header (contains "Vol." or "Supplementary")
            if (text.includes('Vol.') || text.startsWith('Supplementary')) {
                caption.classList.add('volume-header');
                // Also add class to parent for potential styling
                const parent = caption.closest('.nav-item, li');
                if (parent) {
                    parent.classList.add('volume-header-section');
                }
            }
        });
    }

    /**
     * Disable tippy tooltips in navigation sidebars.
     */
    function disableNavTooltips() {
        const containers = document.querySelectorAll(
            '.bd-sidebar-primary, .bd-sidebar-secondary'
        );

        containers.forEach(function(container) {
            const links = container.querySelectorAll('a.reference.internal, a.nav-link');
            links.forEach(function(link) {
                const tippyInstance = link._tippy;
                if (tippyInstance && !tippyInstance.state.isDestroyed) {
                    tippyInstance.destroy();
                }
                if (link.hasAttribute('aria-describedby')) {
                    link.removeAttribute('aria-describedby');
                }
            });
        });
    }

    // Initialize immediately (before DOM ready) to prevent flash
    const isExpert = initExpertMode();

    // Add toggle and apply volume styles when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            insertToggle(createToggleSwitch(isExpert));
            applyVolumeHeaderStyles();
            disableNavTooltips();
        });
    } else {
        insertToggle(createToggleSwitch(isExpert));
        applyVolumeHeaderStyles();
        disableNavTooltips();
    }

    // Run after full load to catch tippy initialization (window.onload).
    window.addEventListener('load', function() {
        setTimeout(disableNavTooltips, 0);
    });
})();
