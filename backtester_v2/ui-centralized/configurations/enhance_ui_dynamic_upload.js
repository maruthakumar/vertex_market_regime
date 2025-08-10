/**
 * Dynamic Excel Upload Enhancement for Multi-File Strategy Configuration
 * 
 * This module enhances the UI to support dynamic file upload based on strategy type
 * and Excel file references. It scans Excel files for *FilePath columns and creates
 * appropriate upload fields dynamically.
 */

class DynamicStrategyUpload {
    constructor() {
        this.strategyConfigs = {
            'tv': {
                name: 'TradingView Strategy',
                files: [
                    { key: 'portfolio', name: 'TV_CONFIG_PORTFOLIO_1.0.0.xlsx', required: true },
                    { key: 'strategy', name: 'TV_CONFIG_STRATEGY_1.0.0.xlsx', required: true },
                    { key: 'signals', name: 'TV_CONFIG_SIGNALS_1.0.0.xlsx', required: false },
                    { key: 'symbols', name: 'TV_CONFIG_SYMBOLS_1.0.0.xlsx', required: false },
                    { key: 'filters', name: 'TV_CONFIG_FILTERS_1.0.0.xlsx', required: false },
                    { key: 'execution', name: 'TV_CONFIG_EXECUTION_1.0.0.xlsx', required: false }
                ]
            },
            'tbs': {
                name: 'Trade Builder Strategy',
                files: [
                    { key: 'portfolio', name: 'TBS_CONFIG_PORTFOLIO_1.0.0.xlsx', required: true },
                    { key: 'strategy', name: 'TBS_CONFIG_STRATEGY_1.0.0.xlsx', required: true }
                ]
            },
            'pos': {
                name: 'Position with Greeks',
                files: [
                    { key: 'portfolio', name: 'POS_CONFIG_PORTFOLIO_1.0.0.xlsx', required: true },
                    { key: 'strategy', name: 'POS_CONFIG_STRATEGY_1.0.0.xlsx', required: true },
                    { key: 'greeks', name: 'POS_CONFIG_GREEKS_1.0.0.xlsx', required: false }
                ]
            },
            'oi': {
                name: 'Open Interest Strategy',
                files: [
                    { key: 'portfolio', name: 'OI_CONFIG_PORTFOLIO_1.0.0.xlsx', required: true },
                    { key: 'strategy', name: 'OI_CONFIG_STRATEGY_1.0.0.xlsx', required: true }
                ]
            },
            'orb': {
                name: 'Opening Range Breakout',
                files: [
                    { key: 'portfolio', name: 'ORB_CONFIG_PORTFOLIO_1.0.0.xlsx', required: true },
                    { key: 'strategy', name: 'ORB_CONFIG_STRATEGY_1.0.0.xlsx', required: true }
                ]
            },
            'ml_indicator': {
                name: 'ML Indicator Strategy',
                files: [
                    { key: 'portfolio', name: 'ML_CONFIG_PORTFOLIO_1.0.0.xlsx', required: true },
                    { key: 'strategy', name: 'ML_CONFIG_STRATEGY_1.0.0.xlsx', required: true },
                    { key: 'indicators', name: 'ML_CONFIG_INDICATORS_1.0.0.xlsx', required: false }
                ]
            },
            'market_regime': {
                name: 'Market Regime Strategy',
                files: [
                    { key: 'portfolio', name: 'MR_CONFIG_PORTFOLIO_1.0.0.xlsx', required: true },
                    { key: 'strategy', name: 'MR_CONFIG_STRATEGY_1.0.0.xlsx', required: true },
                    { key: 'regime', name: 'MR_CONFIG_REGIME_1.0.0.xlsx', required: true },
                    { key: 'optimization', name: 'MR_CONFIG_OPTIMIZATION_1.0.0.xlsx', required: false }
                ]
            }
        };
        
        this.uploadedFiles = {};
        this.dynamicReferences = {};
        this.excelReader = null;
    }
    
    /**
     * Initialize the dynamic upload system
     */
    initialize() {
        // Load Excel parsing library if not already loaded
        if (!window.XLSX) {
            this.loadExcelLibrary().then(() => {
                this.setupEventListeners();
            });
        } else {
            this.setupEventListeners();
        }
    }
    
    /**
     * Load SheetJS library for Excel parsing
     */
    async loadExcelLibrary() {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.sheetjs.com/xlsx-0.20.0/package/dist/xlsx.full.min.js';
            script.onload = () => {
                console.log('✅ Excel parsing library loaded');
                resolve();
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Listen for strategy type changes
        const strategySelect = document.getElementById('strategy-type');
        if (strategySelect) {
            strategySelect.addEventListener('change', (e) => {
                this.handleStrategyChange(e.target.value);
            });
        }
        
        // Listen for tab changes to show file upload section
        const fileTab = document.getElementById('tab-files');
        if (fileTab) {
            fileTab.addEventListener('click', () => {
                this.updateFileUploadSection();
            });
        }
    }
    
    /**
     * Handle strategy type change
     */
    handleStrategyChange(strategyType) {
        console.log(`Strategy changed to: ${strategyType}`);
        
        // Reset uploaded files
        this.uploadedFiles = {};
        this.dynamicReferences = {};
        
        // Update file upload section
        this.updateFileUploadSection();
        
        // Update file count badge
        this.updateFileBadge();
    }
    
    /**
     * Update the file upload section based on selected strategy
     */
    updateFileUploadSection() {
        const fileUploadSection = document.getElementById('fileUploadSection');
        const strategyType = document.getElementById('strategy-type')?.value;
        
        if (!fileUploadSection) return;
        
        if (!strategyType || strategyType === '') {
            fileUploadSection.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i> Please select a strategy type first
                </div>
            `;
            return;
        }
        
        const config = this.strategyConfigs[strategyType];
        if (!config) return;
        
        let html = `
            <div class="backtest-config-section">
                <div class="config-section-header">
                    <h4 class="config-section-title">
                        <i class="fas fa-file-excel"></i>
                        ${config.name} Configuration Files
                    </h4>
                </div>
                <div class="config-section-content">
                    <p class="text-muted mb-4">Upload the Excel configuration files for ${config.name}. Required files are marked with *</p>
                    <div class="upload-grid">
        `;
        
        // Create upload cards for each file
        config.files.forEach(file => {
            html += this.createUploadCard(strategyType, file);
        });
        
        html += `
                    </div>
                    <div id="dynamic-upload-section" class="mt-4">
                        <!-- Dynamic upload fields will be added here based on Excel references -->
                    </div>
                    <div id="upload-validation-results" class="mt-3" style="display: none;">
                        <div class="alert alert-info">
                            <h6><i class="fas fa-info-circle"></i> Validation Results</h6>
                            <div id="validation-details"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        fileUploadSection.innerHTML = html;
        
        // Setup upload handlers
        this.setupUploadHandlers(strategyType);
    }
    
    /**
     * Create an upload card for a file
     */
    createUploadCard(strategyType, file) {
        const uploadId = `${strategyType}-${file.key}-upload`;
        const isUploaded = this.uploadedFiles[uploadId];
        
        return `
            <div class="upload-card ${isUploaded ? 'has-file' : ''}" id="${uploadId}-card" data-strategy="${strategyType}" data-file="${file.key}">
                <div class="upload-icon">
                    <i class="fas fa-file-excel fa-3x text-success"></i>
                </div>
                <div class="upload-title">
                    ${file.name} ${file.required ? '<span class="text-danger">*</span>' : ''}
                </div>
                <div class="upload-description">
                    ${file.required ? 'Required' : 'Optional'} - Drop file here or click to browse
                </div>
                <div class="upload-actions">
                    <input type="file" id="${uploadId}-input" accept=".xlsx,.xls" style="display: none;">
                    <a href="/api/templates/${strategyType}/${file.name}" class="template-link" download>
                        <i class="fas fa-download"></i> Download Template
                    </a>
                </div>
                <div class="file-info" id="${uploadId}-info" style="display: none;"></div>
                <div class="upload-progress" id="${uploadId}-progress" style="display: none;">
                    <div class="progress">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 0%"></div>
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Setup upload handlers for all upload cards
     */
    setupUploadHandlers(strategyType) {
        const config = this.strategyConfigs[strategyType];
        if (!config) return;
        
        config.files.forEach(file => {
            const uploadId = `${strategyType}-${file.key}-upload`;
            const card = document.getElementById(`${uploadId}-card`);
            const input = document.getElementById(`${uploadId}-input`);
            
            if (!card || !input) return;
            
            // Click handler
            card.addEventListener('click', () => {
                input.click();
            });
            
            // Drag and drop handlers
            card.addEventListener('dragover', (e) => {
                e.preventDefault();
                card.classList.add('drag-over');
            });
            
            card.addEventListener('dragleave', () => {
                card.classList.remove('drag-over');
            });
            
            card.addEventListener('drop', (e) => {
                e.preventDefault();
                card.classList.remove('drag-over');
                if (e.dataTransfer.files.length > 0) {
                    this.handleFileUpload(uploadId, e.dataTransfer.files[0]);
                }
            });
            
            // File input change handler
            input.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFileUpload(uploadId, e.target.files[0]);
                }
            });
        });
    }
    
    /**
     * Handle file upload
     */
    async handleFileUpload(uploadId, file) {
        console.log(`Uploading file for ${uploadId}:`, file.name);
        
        // Validate file type
        if (!file.name.endsWith('.xlsx') && !file.name.endsWith('.xls')) {
            this.showNotification('Please upload an Excel file (.xlsx or .xls)', 'error');
            return;
        }
        
        const card = document.getElementById(`${uploadId}-card`);
        const progressDiv = document.getElementById(`${uploadId}-progress`);
        const progressBar = progressDiv?.querySelector('.progress-bar');
        const infoDiv = document.getElementById(`${uploadId}-info`);
        
        // Show progress
        if (progressDiv) {
            progressDiv.style.display = 'block';
            progressBar.style.width = '0%';
        }
        
        try {
            // Read and parse Excel file
            const data = await this.readExcelFile(file);
            
            // Update progress
            if (progressBar) progressBar.style.width = '50%';
            
            // Scan for file references
            const references = await this.scanExcelForReferences(data);
            
            // Update progress
            if (progressBar) progressBar.style.width = '100%';
            
            // Store uploaded file info
            this.uploadedFiles[uploadId] = {
                name: file.name,
                size: file.size,
                data: data,
                references: references
            };
            
            // Update UI
            setTimeout(() => {
                if (progressDiv) progressDiv.style.display = 'none';
                if (card) card.classList.add('has-file');
                if (infoDiv) {
                    infoDiv.style.display = 'block';
                    infoDiv.innerHTML = `
                        <div class="text-success">
                            <i class="fas fa-check-circle"></i> ${file.name} uploaded
                            <br><small>${this.formatFileSize(file.size)} • ${Object.keys(data).length} sheets</small>
                        </div>
                    `;
                }
                
                // Update dynamic references section
                this.updateDynamicReferences();
                
                // Update file badge
                this.updateFileBadge();
            }, 500);
            
        } catch (error) {
            console.error('Error uploading file:', error);
            this.showNotification('Error processing file: ' + error.message, 'error');
            
            if (progressDiv) progressDiv.style.display = 'none';
        }
    }
    
    /**
     * Read Excel file and return parsed data
     */
    async readExcelFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                try {
                    const data = e.target.result;
                    const workbook = XLSX.read(data, { type: 'binary' });
                    
                    const sheets = {};
                    workbook.SheetNames.forEach(sheetName => {
                        sheets[sheetName] = XLSX.utils.sheet_to_json(
                            workbook.Sheets[sheetName],
                            { header: 1 }
                        );
                    });
                    
                    resolve(sheets);
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = reject;
            reader.readAsBinaryString(file);
        });
    }
    
    /**
     * Scan Excel data for file path references
     */
    async scanExcelForReferences(data) {
        const references = [];
        
        // Look for columns ending with 'FilePath' or 'ExcelFilePath'
        Object.entries(data).forEach(([sheetName, sheetData]) => {
            if (!Array.isArray(sheetData) || sheetData.length < 2) return;
            
            const headers = sheetData[0];
            if (!Array.isArray(headers)) return;
            
            headers.forEach((header, colIndex) => {
                if (typeof header === 'string' && 
                    (header.endsWith('FilePath') || header.endsWith('ExcelFilePath'))) {
                    
                    // Get all values in this column
                    for (let rowIndex = 1; rowIndex < sheetData.length; rowIndex++) {
                        const value = sheetData[rowIndex][colIndex];
                        if (value && typeof value === 'string' && value.trim()) {
                            references.push({
                                sheet: sheetName,
                                column: header,
                                row: rowIndex + 1,
                                path: value.trim(),
                                type: this.inferFileType(header, value)
                            });
                        }
                    }
                }
            });
        });
        
        return references;
    }
    
    /**
     * Infer file type from column name and file path
     */
    inferFileType(columnName, filePath) {
        const lowerColumn = columnName.toLowerCase();
        const lowerPath = filePath.toLowerCase();
        
        if (lowerColumn.includes('indicator') || lowerPath.includes('indicator')) {
            return 'indicators';
        } else if (lowerColumn.includes('symbol') || lowerPath.includes('symbol')) {
            return 'symbols';
        } else if (lowerColumn.includes('signal') || lowerPath.includes('signal')) {
            return 'signals';
        } else if (lowerColumn.includes('filter') || lowerPath.includes('filter')) {
            return 'filters';
        } else if (lowerColumn.includes('regime') || lowerPath.includes('regime')) {
            return 'regime';
        } else if (lowerColumn.includes('greek') || lowerPath.includes('greek')) {
            return 'greeks';
        } else {
            return 'additional';
        }
    }
    
    /**
     * Update dynamic references section
     */
    updateDynamicReferences() {
        const dynamicSection = document.getElementById('dynamic-upload-section');
        if (!dynamicSection) return;
        
        // Collect all references from uploaded files
        const allReferences = [];
        Object.values(this.uploadedFiles).forEach(file => {
            if (file.references) {
                allReferences.push(...file.references);
            }
        });
        
        if (allReferences.length === 0) {
            dynamicSection.innerHTML = '';
            return;
        }
        
        // Group references by type
        const groupedRefs = {};
        allReferences.forEach(ref => {
            if (!groupedRefs[ref.type]) {
                groupedRefs[ref.type] = [];
            }
            groupedRefs[ref.type].push(ref);
        });
        
        let html = `
            <div class="alert alert-info mb-3">
                <i class="fas fa-info-circle"></i> Additional files referenced in Excel configuration:
            </div>
            <div class="row">
        `;
        
        Object.entries(groupedRefs).forEach(([type, refs]) => {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="upload-card dynamic-upload" data-ref-type="${type}">
                        <div class="upload-icon">
                            <i class="fas fa-file-alt fa-2x text-primary"></i>
                        </div>
                        <div class="upload-title">
                            ${this.getTypeDisplayName(type)} Files
                        </div>
                        <div class="upload-description">
                            Referenced in: ${refs.map(r => r.column).join(', ')}
                        </div>
                        <div class="upload-actions">
                            <input type="file" id="dynamic-${type}-input" accept=".xlsx,.xls,.csv" style="display: none;">
                            <button class="btn btn-sm btn-primary" onclick="document.getElementById('dynamic-${type}-input').click()">
                                <i class="fas fa-upload"></i> Upload
                            </button>
                        </div>
                        <div class="file-info" id="dynamic-${type}-info" style="display: none;"></div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        dynamicSection.innerHTML = html;
        
        // Setup handlers for dynamic uploads
        Object.keys(groupedRefs).forEach(type => {
            const input = document.getElementById(`dynamic-${type}-input`);
            if (input) {
                input.addEventListener('change', (e) => {
                    if (e.target.files.length > 0) {
                        this.handleDynamicFileUpload(type, e.target.files[0]);
                    }
                });
            }
        });
    }
    
    /**
     * Get display name for file type
     */
    getTypeDisplayName(type) {
        const names = {
            'indicators': 'Indicator Configuration',
            'symbols': 'Symbol List',
            'signals': 'Signal Configuration',
            'filters': 'Filter Rules',
            'regime': 'Market Regime',
            'greeks': 'Greeks Configuration',
            'additional': 'Additional Configuration'
        };
        return names[type] || 'Configuration';
    }
    
    /**
     * Handle dynamic file upload
     */
    async handleDynamicFileUpload(type, file) {
        console.log(`Uploading dynamic file for ${type}:`, file.name);
        
        const infoDiv = document.getElementById(`dynamic-${type}-info`);
        
        this.dynamicReferences[type] = {
            name: file.name,
            size: file.size
        };
        
        if (infoDiv) {
            infoDiv.style.display = 'block';
            infoDiv.innerHTML = `
                <div class="text-success mt-2">
                    <i class="fas fa-check-circle"></i> ${file.name}
                    <br><small>${this.formatFileSize(file.size)}</small>
                </div>
            `;
        }
        
        this.showNotification(`${this.getTypeDisplayName(type)} file uploaded successfully`, 'success');
    }
    
    /**
     * Update file count badge
     */
    updateFileBadge() {
        const badge = document.getElementById('files-badge');
        if (!badge) return;
        
        const uploadedCount = Object.keys(this.uploadedFiles).length + 
                              Object.keys(this.dynamicReferences).length;
        
        if (uploadedCount > 0) {
            badge.style.display = 'inline';
            badge.textContent = uploadedCount;
        } else {
            badge.style.display = 'none';
        }
    }
    
    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
    
    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        // Use existing notification system if available
        if (window.showNotification) {
            window.showNotification(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
    
    /**
     * Get all uploaded files for submission
     */
    getUploadedFiles() {
        return {
            strategy: document.getElementById('strategy-type')?.value,
            files: this.uploadedFiles,
            dynamicFiles: this.dynamicReferences
        };
    }
}

// Initialize on DOM content loaded
document.addEventListener('DOMContentLoaded', function() {
    window.dynamicStrategyUpload = new DynamicStrategyUpload();
    window.dynamicStrategyUpload.initialize();
    console.log('✅ Dynamic Strategy Upload system initialized');
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DynamicStrategyUpload;
}