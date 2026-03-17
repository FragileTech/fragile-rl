/**
 * Prompt download menu for Fragile Docs.
 * Adds a top-right dropdown with volume + proofs + format selection.
 */
(function() {
    'use strict';

    const VOLUMES = [
        { id: '1_agent', label: 'Vol 1 - Agent' },
        { id: '2_hypostructure', label: 'Vol 2 - Hypostructure' },
        { id: '3_fractal_gas', label: 'Vol 3 - Fractal Gas' },
    ];

    const PROOFS = [
        { id: 'with', label: 'With proofs', slug: 'with-proofs' },
        { id: 'without', label: 'Without proofs', slug: 'no-proofs' },
    ];

    const FORMATS = [
        { id: 'md', label: 'Markdown' },
        { id: 'txt', label: 'Text' },
    ];

    function getContentRoot() {
        const root = document.documentElement.dataset.content_root;
        return root || './';
    }

    function buildFilename(volumeId, proofsId, format) {
        const proof = PROOFS.find(item => item.id === proofsId) || PROOFS[0];
        return volumeId + '-' + proof.slug + '.' + format;
    }

    function buildHref(volumeId, proofsId, format) {
        const filename = buildFilename(volumeId, proofsId, format);
        return getContentRoot() + '_static/prompts/' + filename;
    }

    function createMenu() {
        const container = document.createElement('div');
        container.className = 'prompt-download-container';

        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'prompt-download-button';
        button.setAttribute('aria-haspopup', 'true');
        button.setAttribute('aria-expanded', 'false');
        button.setAttribute('title', 'Download prompt files');

        const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        icon.setAttribute('viewBox', '0 0 24 24');
        icon.setAttribute('width', '14');
        icon.setAttribute('height', '14');
        icon.setAttribute('aria-hidden', 'true');
        icon.classList.add('prompt-download-icon');

        const iconPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        iconPath.setAttribute('d', 'M12 3v10m0 0l4-4m-4 4l-4-4M4 17v3h16v-3');
        iconPath.setAttribute('fill', 'none');
        iconPath.setAttribute('stroke', 'currentColor');
        iconPath.setAttribute('stroke-width', '1.6');
        iconPath.setAttribute('stroke-linecap', 'round');
        iconPath.setAttribute('stroke-linejoin', 'round');
        icon.appendChild(iconPath);

        const buttonText = document.createElement('span');
        buttonText.textContent = 'Download prompt';

        button.appendChild(icon);
        button.appendChild(buttonText);

        const menu = document.createElement('div');
        menu.className = 'prompt-download-menu';
        menu.setAttribute('role', 'menu');

        const title = document.createElement('div');
        title.className = 'prompt-download-title';
        title.textContent = 'Prompt downloads';
        menu.appendChild(title);

        const volumeRow = document.createElement('div');
        volumeRow.className = 'prompt-download-row';

        const volumeLabel = document.createElement('label');
        volumeLabel.className = 'prompt-download-label';
        volumeLabel.textContent = 'Volume';

        const volumeSelect = document.createElement('select');
        volumeSelect.className = 'prompt-download-select';
        volumeSelect.setAttribute('aria-label', 'Select volume');

        VOLUMES.forEach(function(volume) {
            const option = document.createElement('option');
            option.value = volume.id;
            option.textContent = volume.label;
            volumeSelect.appendChild(option);
        });

        volumeRow.appendChild(volumeLabel);
        volumeRow.appendChild(volumeSelect);
        menu.appendChild(volumeRow);

        const proofsRow = document.createElement('div');
        proofsRow.className = 'prompt-download-row';

        const proofsLabel = document.createElement('div');
        proofsLabel.className = 'prompt-download-label';
        proofsLabel.textContent = 'Proofs';

        const proofsToggle = document.createElement('div');
        proofsToggle.className = 'prompt-download-toggle';

        const proofButtons = PROOFS.map(function(proof) {
            const proofButton = document.createElement('button');
            proofButton.type = 'button';
            proofButton.className = 'prompt-download-toggle-button';
            proofButton.textContent = proof.label;
            proofButton.dataset.proof = proof.id;
            proofButton.setAttribute('aria-pressed', proof.id === 'with' ? 'true' : 'false');
            proofsToggle.appendChild(proofButton);
            return proofButton;
        });

        proofsRow.appendChild(proofsLabel);
        proofsRow.appendChild(proofsToggle);
        menu.appendChild(proofsRow);

        const formatRow = document.createElement('div');
        formatRow.className = 'prompt-download-row';

        const formatLabel = document.createElement('div');
        formatLabel.className = 'prompt-download-label';
        formatLabel.textContent = 'Format';

        const formatLinks = document.createElement('div');
        formatLinks.className = 'prompt-download-formats';

        const formatButtons = FORMATS.map(function(format) {
            const link = document.createElement('a');
            link.className = 'prompt-download-link';
            link.textContent = format.label;
            link.dataset.format = format.id;
            link.setAttribute('role', 'menuitem');
            link.setAttribute('download', '');
            formatLinks.appendChild(link);
            return link;
        });

        formatRow.appendChild(formatLabel);
        formatRow.appendChild(formatLinks);
        menu.appendChild(formatRow);

        container.appendChild(button);
        container.appendChild(menu);

        let selectedProof = 'with';
        let selectedVolume = VOLUMES[0].id;

        function updateLinks() {
            formatButtons.forEach(function(link) {
                const format = link.dataset.format;
                const filename = buildFilename(selectedVolume, selectedProof, format);
                link.href = buildHref(selectedVolume, selectedProof, format);
                link.setAttribute('download', filename);
            });

            proofButtons.forEach(function(btn) {
                const isActive = btn.dataset.proof === selectedProof;
                btn.classList.toggle('is-active', isActive);
                btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
            });
        }

        function setMenuOpen(open) {
            container.classList.toggle('is-open', open);
            button.setAttribute('aria-expanded', open ? 'true' : 'false');
        }

        button.addEventListener('click', function(event) {
            event.preventDefault();
            event.stopPropagation();
            setMenuOpen(!container.classList.contains('is-open'));
        });

        volumeSelect.addEventListener('change', function() {
            selectedVolume = volumeSelect.value;
            updateLinks();
        });

        proofButtons.forEach(function(btn) {
            btn.addEventListener('click', function() {
                selectedProof = btn.dataset.proof || 'with';
                updateLinks();
            });
        });

        formatButtons.forEach(function(link) {
            link.addEventListener('click', function() {
                setMenuOpen(false);
            });
        });

        document.addEventListener('click', function(event) {
            if (!container.contains(event.target)) {
                setMenuOpen(false);
            }
        });

        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                setMenuOpen(false);
            }
        });

        updateLinks();

        return container;
    }

    function insertMenu(menu) {
        const headerItems = document.querySelector('.header-article-items');
        if (headerItems) {
            headerItems.appendChild(menu);
            return;
        }

        const primaryToggle = document.querySelector('.sidebar-toggle.primary-toggle');
        if (primaryToggle) {
            const wrapper = primaryToggle.closest('.header-article-item');
            if (wrapper && wrapper.parentNode) {
                wrapper.parentNode.insertBefore(menu, wrapper.nextSibling);
                return;
            }
        }

        menu.classList.add('prompt-download-container-fixed');
        document.body.appendChild(menu);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            insertMenu(createMenu());
        });
    } else {
        insertMenu(createMenu());
    }
})();
