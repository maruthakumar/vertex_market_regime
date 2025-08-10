/**
 * Integration with Existing Excel Upload System
 * 
 * This module enhances the existing UI to work with the current Excel structure
 * where PortfolioSetting files contain StrategySetting sheets that reference
 * other Excel files via StrategyExcelFilePath column.
 */

class ExistingExcelUploadIntegration {
    constructor() {
        this.strategyPortfolioMap = {
            'tv': { portfolio: 'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx', expectedFiles: 2 },
            'tbs': { portfolio: 'TBS_CONFIG_PORTFOLIO_1.0.0.xlsx', expectedFiles: 2 },
            'pos': { portfolio: 'POS_CONFIG_PORTFOLIO_1.0.0.xlsx', expectedFiles: 3 },
            'oi': { portfolio: 'OI_CONFIG_PORTFOLIO_1.0.0.xlsx', expectedFiles: 2 },
            'orb': { portfolio: 'ORB_CONFIG_PORTFOLIO_1.0.0.xlsx', expectedFiles: 2 },
            'ml_indicator': { portfolio: 'ML_CONFIG_PORTFOLIO_1.0.0.xlsx', expectedFiles: 3 },
            'market_regime': { portfolio: 'MR_CONFIG_PORTFOLIO_1.0.0.xlsx', expectedFiles: 4 }
        };
        
        this.uploadedFiles = new Map();
        this.requiredFiles = new Map();
    }
    
    /**
     * Update the existing file upload section to handle portfolio-based uploads
     */
    updateFileUploadSection() {
        const fileUploadSection = document.getElementById('fileUploadSection');
        const strategyType = document.getElementById('strategy-type')?.value;
        
        if (!fileUploadSection || !strategyType) return;
        
        const config = this.strategyPortfolioMap[strategyType];
        if (!config) {
            fileUploadSection.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i> Unknown strategy type: ${strategyType}
                </div>
            `;
            return;
        }
        
        let html = `
            <div class="backtest-config-section">
                <div class="config-section-header">
                    <h4 class="config-section-title">
                        <i class="fas fa-file-excel"></i>
                        Upload Configuration Files
                    </h4>
                </div>
                <div class="config-section-content">
                    <div class="row">
                        <div class="col-md-12">
                            <div class="alert alert-info mb-4">
                                <i class="fas fa-info-circle"></i> 
                                <strong>Step 1:</strong> Upload the Portfolio configuration file first.
                                The system will automatically detect required additional files from the StrategySetting sheet.
                            </div>
                        </div>
                    </div>
                    
                    <!-- Portfolio File Upload -->
                    <div class="upload-card primary-upload" id="portfolio-upload-card" data-strategy="${strategyType}">
                        <div class="upload-icon">
                            <i class="fas fa-file-excel fa-3x text-primary"></i>
                        </div>
                        <div class="upload-title">
                            Portfolio Configuration <span class="text-danger">*</span>
                        </div>
                        <div class="upload-description">
                            Upload ${config.portfolio} or similar portfolio file
                        </div>
                        <div class="upload-actions">
                            <input type="file" id="portfolio-file-input" accept=".xlsx,.xls" style="display: none;">
                            <button class="btn btn-primary btn-sm" onclick="document.getElementById('portfolio-file-input').click()">
                                <i class="fas fa-upload"></i> Select File
                            </button>
                        </div>
                        <div class="file-info" id="portfolio-file-info" style="display: none;"></div>
                    </div>
                    
                    <!-- Dynamic file requirements will be shown here -->
                    <div id="dynamic-file-requirements" style="display: none;" class="mt-4">
                        <h5 class="mb-3">
                            <i class="fas fa-list-check"></i> Required Files Detected
                        </h5>
                        <div id="required-files-list" class="row">
                            <!-- Dynamically populated based on StrategySetting sheet -->
                        </div>
                    </div>
                    
                    <!-- Validation Summary -->
                    <div id="validation-summary" style="display: none;" class="mt-4">
                        <div class="alert alert-success">
                            <h6><i class="fas fa-check-circle"></i> Configuration Status</h6>
                            <div id="validation-details"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        fileUploadSection.innerHTML = html;
        this.setupPortfolioUploadHandler();
    }
    
    /**
     * Setup handler for portfolio file upload
     */
    setupPortfolioUploadHandler() {
        const portfolioInput = document.getElementById('portfolio-file-input');
        const portfolioCard = document.getElementById('portfolio-upload-card');
        
        if (!portfolioInput || !portfolioCard) return;
        
        // File input handler
        portfolioInput.addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                await this.handlePortfolioUpload(e.target.files[0]);
            }
        });
        
        // Drag and drop
        portfolioCard.addEventListener('dragover', (e) => {
            e.preventDefault();
            portfolioCard.classList.add('drag-over');
        });
        
        portfolioCard.addEventListener('dragleave', () => {
            portfolioCard.classList.remove('drag-over');
        });
        
        portfolioCard.addEventListener('drop', async (e) => {
            e.preventDefault();
            portfolioCard.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                await this.handlePortfolioUpload(e.dataTransfer.files[0]);
            }
        });
    }
    
    /**
     * Handle portfolio file upload and scan for references
     */
    async handlePortfolioUpload(file) {
        const strategyType = document.getElementById('strategy-type')?.value;
        
        try {
            // Show uploading state
            const portfolioInfo = document.getElementById('portfolio-file-info');
            if (portfolioInfo) {
                portfolioInfo.style.display = 'block';
                portfolioInfo.innerHTML = `
                    <div class="text-info">
                        <i class="fas fa-spinner fa-spin"></i> Processing ${file.name}...
                    </div>
                `;
            }
            
            // Read Excel file
            const data = await this.readExcelFile(file);
            
            // Extract StrategySetting sheet
            const strategySettingData = this.extractStrategySettings(data);
            
            if (!strategySettingData || strategySettingData.length === 0) {
                throw new Error('No StrategySetting sheet found or sheet is empty');
            }
            
            // Store portfolio file
            this.uploadedFiles.set('portfolio', {
                name: file.name,
                size: file.size,
                data: data
            });
            
            // Update UI with success
            if (portfolioInfo) {
                portfolioInfo.innerHTML = `
                    <div class="text-success">
                        <i class="fas fa-check-circle"></i> ${file.name} uploaded
                        <br><small>${this.formatFileSize(file.size)} • ${Object.keys(data).length} sheets</small>
                    </div>
                `;
            }
            
            // Show required files based on StrategySetting
            this.showRequiredFiles(strategySettingData);
            
        } catch (error) {
            console.error('Error processing portfolio file:', error);
            this.showNotification('Error: ' + error.message, 'error');
            
            const portfolioInfo = document.getElementById('portfolio-file-info');
            if (portfolioInfo) {
                portfolioInfo.innerHTML = `
                    <div class="text-danger">
                        <i class="fas fa-exclamation-circle"></i> Error: ${error.message}
                    </div>
                `;
            }
        }
    }
    
    /**
     * Extract StrategySetting data from Excel sheets
     */
    extractStrategySettings(excelData) {
        // Look for StrategySetting sheet
        const settingsSheet = excelData['StrategySetting'] || 
                            excelData['Strategy Setting'] || 
                            excelData['StrategySettings'];
        
        if (!settingsSheet || !Array.isArray(settingsSheet) || settingsSheet.length < 2) {
            return null;
        }
        
        // Parse sheet data
        const headers = settingsSheet[0];
        const rows = settingsSheet.slice(1);
        
        // Find column indices
        const nameIndex = headers.findIndex(h => 
            h && h.toString().toLowerCase().includes('strategyname'));
        const pathIndex = headers.findIndex(h => 
            h && h.toString().toLowerCase().includes('strategyexcelfilepath'));
        const enabledIndex = headers.findIndex(h => 
            h && h.toString().toLowerCase().includes('enabled'));
        
        if (nameIndex === -1 || pathIndex === -1) {
            throw new Error('Required columns not found in StrategySetting sheet');
        }
        
        // Extract settings
        const settings = rows
            .filter(row => row[nameIndex] && row[pathIndex])
            .map(row => ({
                name: row[nameIndex],
                filePath: row[pathIndex],
                enabled: enabledIndex !== -1 ? row[enabledIndex] : true
            }));
        
        return settings;
    }
    
    /**
     * Show required files based on StrategySetting data
     */
    showRequiredFiles(strategySettings) {
        const requirementsDiv = document.getElementById('dynamic-file-requirements');
        const filesList = document.getElementById('required-files-list');
        
        if (!requirementsDiv || !filesList) return;
        
        requirementsDiv.style.display = 'block';
        
        // Clear required files
        this.requiredFiles.clear();
        
        // Generate upload cards for each required file
        let html = '';
        strategySettings.forEach((setting, index) => {
            const fileId = `strategy-file-${index}`;
            this.requiredFiles.set(fileId, setting);
            
            html += `
                <div class="col-md-6 mb-3">
                    <div class="upload-card ${setting.enabled ? '' : 'disabled'}" 
                         id="${fileId}-card" 
                         data-file-path="${setting.filePath}">
                        <div class="upload-icon">
                            <i class="fas fa-file-excel fa-2x ${setting.enabled ? 'text-success' : 'text-muted'}"></i>
                        </div>
                        <div class="upload-title">
                            ${setting.name}
                        </div>
                        <div class="upload-description">
                            ${setting.filePath}
                            ${!setting.enabled ? '<br><small class="text-muted">(Disabled in config)</small>' : ''}
                        </div>
                        ${setting.enabled ? `
                            <div class="upload-actions">
                                <input type="file" id="${fileId}-input" accept=".xlsx,.xls" style="display: none;">
                                <button class="btn btn-sm btn-outline-primary" onclick="document.getElementById('${fileId}-input').click()">
                                    <i class="fas fa-upload"></i> Upload
                                </button>
                            </div>
                            <div class="file-info" id="${fileId}-info" style="display: none;"></div>
                        ` : ''}
                    </div>
                </div>
            `;
        });
        
        filesList.innerHTML = html;
        
        // Setup handlers for required files
        strategySettings.forEach((setting, index) => {
            if (setting.enabled) {
                this.setupRequiredFileHandler(`strategy-file-${index}`, setting);
            }
        });
        
        // Update validation summary
        this.updateValidationSummary();
    }
    
    /**
     * Setup handler for required file upload
     */
    setupRequiredFileHandler(fileId, setting) {
        const input = document.getElementById(`${fileId}-input`);
        const card = document.getElementById(`${fileId}-card`);
        
        if (!input || !card) return;
        
        input.addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                await this.handleRequiredFileUpload(fileId, setting, e.target.files[0]);
            }
        });
        
        // Drag and drop
        card.addEventListener('dragover', (e) => {
            e.preventDefault();
            card.classList.add('drag-over');
        });
        
        card.addEventListener('dragleave', () => {
            card.classList.remove('drag-over');
        });
        
        card.addEventListener('drop', async (e) => {
            e.preventDefault();
            card.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                await this.handleRequiredFileUpload(fileId, setting, e.dataTransfer.files[0]);
            }
        });
    }
    
    /**
     * Handle required file upload
     */
    async handleRequiredFileUpload(fileId, setting, file) {
        const infoDiv = document.getElementById(`${fileId}-info`);
        const card = document.getElementById(`${fileId}-card`);
        
        try {
            // Validate file name matches expected
            if (!file.name.includes(setting.filePath.split('/').pop().split('.')[0])) {
                console.warn(`File name ${file.name} may not match expected ${setting.filePath}`);
            }
            
            // Store file
            this.uploadedFiles.set(fileId, {
                name: file.name,
                size: file.size,
                setting: setting
            });
            
            // Update UI
            if (infoDiv) {
                infoDiv.style.display = 'block';
                infoDiv.innerHTML = `
                    <div class="text-success mt-2">
                        <i class="fas fa-check-circle"></i> ${file.name}
                        <br><small>${this.formatFileSize(file.size)}</small>
                    </div>
                `;
            }
            
            if (card) {
                card.classList.add('has-file');
            }
            
            // Update validation summary
            this.updateValidationSummary();
            
        } catch (error) {
            console.error('Error uploading file:', error);
            if (infoDiv) {
                infoDiv.innerHTML = `
                    <div class="text-danger mt-2">
                        <i class="fas fa-exclamation-circle"></i> Error: ${error.message}
                    </div>
                `;
            }
        }
    }
    
    /**
     * Update validation summary
     */
    updateValidationSummary() {
        const summaryDiv = document.getElementById('validation-summary');
        const detailsDiv = document.getElementById('validation-details');
        
        if (!summaryDiv || !detailsDiv) return;
        
        const requiredCount = Array.from(this.requiredFiles.values())
            .filter(s => s.enabled).length + 1; // +1 for portfolio
        const uploadedCount = this.uploadedFiles.size;
        
        if (uploadedCount > 0) {
            summaryDiv.style.display = 'block';
            
            const isComplete = uploadedCount === requiredCount;
            summaryDiv.className = isComplete ? 'alert alert-success' : 'alert alert-warning';
            
            detailsDiv.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <strong>Files Uploaded:</strong> ${uploadedCount} / ${requiredCount}
                    </div>
                    <div class="col-md-6">
                        <strong>Status:</strong> 
                        ${isComplete ? 
                            '<span class="text-success"><i class="fas fa-check"></i> Ready to process</span>' : 
                            '<span class="text-warning"><i class="fas fa-hourglass-half"></i> Waiting for files</span>'}
                    </div>
                </div>
                ${isComplete ? `
                    <div class="mt-3">
                        <button class="btn btn-success" onclick="window.existingExcelUpload.processFiles()">
                            <i class="fas fa-play"></i> Start Backtest
                        </button>
                    </div>
                ` : ''}
            `;
        }
        
        // Update file count badge
        const badge = document.getElementById('files-badge');
        if (badge) {
            badge.style.display = uploadedCount > 0 ? 'inline' : 'none';
            badge.textContent = uploadedCount;
        }
    }
    
    /**
     * Read Excel file using SheetJS
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
     * Process all uploaded files
     */
    async processFiles() {
        console.log('Processing files:', this.uploadedFiles);
        
        // Prepare form data
        const formData = new FormData();
        formData.append('strategy_type', document.getElementById('strategy-type').value);
        
        // Add all uploaded files
        this.uploadedFiles.forEach((fileInfo, key) => {
            if (fileInfo.file) {
                formData.append(key, fileInfo.file);
            }
        });
        
        try {
            const response = await fetch('/api/v1/backtest/start', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            if (result.success) {
                this.showNotification('Backtest started successfully!', 'success');
                // Redirect to results or monitoring page
                window.location.hash = '#results';
            } else {
                throw new Error(result.message || 'Failed to start backtest');
            }
        } catch (error) {
            this.showNotification('Error: ' + error.message, 'error');
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
        if (window.showNotification) {
            window.showNotification(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
    
    /**
     * Initialize integration
     */
    initialize() {
        // Listen for strategy changes
        const strategySelect = document.getElementById('strategy-type');
        if (strategySelect) {
            strategySelect.addEventListener('change', () => {
                this.uploadedFiles.clear();
                this.requiredFiles.clear();
                this.updateFileUploadSection();
            });
        }
        
        // Listen for file upload tab
        const fileTab = document.getElementById('tab-files');
        if (fileTab) {
            fileTab.addEventListener('click', () => {
                this.updateFileUploadSection();
            });
        }
        
        console.log('✅ Existing Excel Upload Integration initialized');
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    // Load SheetJS if not already loaded
    if (!window.XLSX) {
        const script = document.createElement('script');
        script.src = 'https://cdn.sheetjs.com/xlsx-0.20.0/package/dist/xlsx.full.min.js';
        script.onload = () => {
            window.existingExcelUpload = new ExistingExcelUploadIntegration();
            window.existingExcelUpload.initialize();
        };
        document.head.appendChild(script);
    } else {
        window.existingExcelUpload = new ExistingExcelUploadIntegration();
        window.existingExcelUpload.initialize();
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ExistingExcelUploadIntegration;
}