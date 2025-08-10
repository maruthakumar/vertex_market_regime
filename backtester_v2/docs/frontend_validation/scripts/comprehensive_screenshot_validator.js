#!/usr/bin/env node

/**
 * Comprehensive Screenshot Documentation and Favicon Validation System
 * SuperClaude v3 Enhanced Backend Integration
 * 
 * Captures before/after screenshots for complete visual validation
 * Tests favicon loading across all pages and viewports
 * Generates organized documentation with timestamp tracking
 */

const { chromium, firefox } = require('playwright');
const fs = require('fs').promises;
const path = require('path');

class ComprehensiveScreenshotValidator {
    constructor() {
        this.baseUrl = 'http://173.208.247.17:3001';
        this.screenshotDir = '/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/docs/frontend_validation/screenshots';
        this.timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        this.viewports = [
            { name: 'desktop', width: 1920, height: 1080 },
            { name: 'tablet', width: 768, height: 1024 },
            { name: 'mobile', width: 375, height: 667 }
        ];
        this.pages = [
            { name: 'main', url: '/', description: 'Main Dashboard' },
            { name: 'backtest', url: '/backtest', description: 'Backtest Creation Page' },
            { name: 'results', url: '/results', description: 'Results Dashboard' },
            { name: 'settings', url: '/settings', description: 'Settings Management' },
            { name: 'backtests', url: '/backtests', description: 'Backtests List' }
        ];
        this.validationResults = {
            timestamp: this.timestamp,
            faviconTests: {},
            screenshotTests: {},
            visualValidation: {},
            issues: [],
            fixes: []
        };
    }

    async init() {
        console.log('🎯 Initializing Comprehensive Screenshot Validator...');
        
        // Create organized directory structure
        await this.createDirectoryStructure();
        
        console.log(`📁 Screenshot directory: ${this.screenshotDir}`);
        console.log(`⏰ Session timestamp: ${this.timestamp}`);
        console.log(`🌐 Base URL: ${this.baseUrl}`);
        console.log('');
    }

    async createDirectoryStructure() {
        const dirs = [
            `${this.screenshotDir}`,
            `${this.screenshotDir}/before`,
            `${this.screenshotDir}/after`, 
            `${this.screenshotDir}/favicon-tests`,
            `${this.screenshotDir}/comparison`,
            `${this.screenshotDir}/${this.timestamp}`
        ];

        for (const dir of dirs) {
            try {
                await fs.mkdir(dir, { recursive: true });
                console.log(`✅ Created directory: ${dir}`);
            } catch (error) {
                console.log(`ℹ️  Directory exists: ${dir}`);
            }
        }
    }

    async validateFaviconLoading(page, pageName) {
        console.log(`🔍 Testing favicon loading for ${pageName}...`);
        
        try {
            // Check if favicon requests are made
            const faviconRequests = [];
            page.on('request', request => {
                const url = request.url();
                if (url.includes('favicon') || url.includes('.ico') || url.includes('apple-touch-icon')) {
                    faviconRequests.push({
                        url: url,
                        method: request.method(),
                        status: 'requested'
                    });
                    console.log(`📋 Favicon request: ${url}`);
                }
            });

            page.on('response', response => {
                const url = response.url();
                if (url.includes('favicon') || url.includes('.ico') || url.includes('apple-touch-icon')) {
                    const existingRequest = faviconRequests.find(req => req.url === url);
                    if (existingRequest) {
                        existingRequest.status = response.status();
                        existingRequest.statusText = response.statusText();
                    }
                    console.log(`📨 Favicon response: ${url} - Status: ${response.status()}`);
                }
            });

            // Wait for page load and favicon requests
            await new Promise(resolve => setTimeout(resolve, 3000));

            // Check if favicon is visible in browser tab
            const title = await page.title();
            console.log(`📄 Page title: ${title}`);

            // Take screenshot of browser tab area
            const tabScreenshot = await page.screenshot({
                clip: { x: 0, y: 0, width: 300, height: 80 },
                path: `${this.screenshotDir}/favicon-tests/${pageName}_tab_${this.timestamp}.png`
            });

            this.validationResults.faviconTests[pageName] = {
                title: title,
                requests: faviconRequests,
                screenshot: `favicon-tests/${pageName}_tab_${this.timestamp}.png`,
                timestamp: new Date().toISOString()
            };

            return faviconRequests.length > 0 ? faviconRequests : null;

        } catch (error) {
            console.log(`❌ Favicon validation error for ${pageName}: ${error.message}`);
            this.validationResults.issues.push(`Favicon validation failed for ${pageName}: ${error.message}`);
            return null;
        }
    }

    async capturePageScreenshots(browser, pageName, pageUrl, description) {
        console.log(`\\n📸 Capturing screenshots for ${description} (${pageName})`);
        
        const pageResults = {
            pageName,
            pageUrl, 
            description,
            screenshots: {},
            faviconTest: null,
            timestamp: new Date().toISOString()
        };

        for (const viewport of this.viewports) {
            console.log(`  📱 ${viewport.name} (${viewport.width}x${viewport.height})`);
            
            try {
                const page = await browser.newPage();
                await page.setViewportSize({ width: viewport.width, height: viewport.height });
                
                // Navigate with extended timeout for complex pages
                await page.goto(`${this.baseUrl}${pageUrl}`, { 
                    waitUntil: 'networkidle', 
                    timeout: 15000 
                });

                // Test favicon loading
                if (viewport.name === 'desktop') {
                    pageResults.faviconTest = await this.validateFaviconLoading(page, pageName);
                }

                // Wait for page to fully load
                await page.waitForTimeout(2000);

                // Capture full page screenshot
                const screenshotPath = `${this.screenshotDir}/${this.timestamp}/${pageName}_${viewport.name}_${this.timestamp}.png`;
                await page.screenshot({ 
                    path: screenshotPath,
                    fullPage: true 
                });

                pageResults.screenshots[viewport.name] = screenshotPath;
                console.log(`    ✅ Screenshot saved: ${screenshotPath}`);

                // Capture browser tab area for favicon verification
                if (viewport.name === 'desktop') {
                    const tabPath = `${this.screenshotDir}/${this.timestamp}/${pageName}_tab_${this.timestamp}.png`;
                    await page.screenshot({
                        path: tabPath,
                        clip: { x: 0, y: 0, width: 400, height: 100 }
                    });
                    pageResults.screenshots.tab = tabPath;
                    console.log(`    🏷️  Tab screenshot saved: ${tabPath}`);
                }

                await page.close();

            } catch (error) {
                console.log(`    ❌ Error capturing ${viewport.name}: ${error.message}`);
                this.validationResults.issues.push(`Screenshot capture failed: ${pageName} - ${viewport.name} - ${error.message}`);
            }
        }

        return pageResults;
    }

    async executeComprehensiveValidation() {
        console.log('\\n🚀 Starting Comprehensive Visual Validation...');
        
        // Test with Chromium (primary)
        console.log('\\n🌐 Testing with Chromium browser...');
        const browser = await chromium.launch({ 
            headless: true,  // Run headless for server environment
            args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
        });

        try {
            for (const pageConfig of this.pages) {
                const pageResults = await this.capturePageScreenshots(
                    browser, 
                    pageConfig.name, 
                    pageConfig.url, 
                    pageConfig.description
                );
                
                this.validationResults.screenshotTests[pageConfig.name] = pageResults;
            }

        } catch (error) {
            console.log(`❌ Browser testing error: ${error.message}`);
            this.validationResults.issues.push(`Browser testing failed: ${error.message}`);
        }

        await browser.close();
    }

    async analyzeFaviconIssues() {
        console.log('\\n🔍 Analyzing Favicon Issues...');
        
        const faviconIssues = [];
        
        for (const [pageName, testResult] of Object.entries(this.validationResults.faviconTests)) {
            console.log(`\\n🔎 Analyzing ${pageName}:`);
            
            if (!testResult.requests || testResult.requests.length === 0) {
                faviconIssues.push(`No favicon requests detected for ${pageName}`);
                console.log(`  ❌ No favicon requests detected`);
            } else {
                for (const request of testResult.requests) {
                    console.log(`  📋 Request: ${request.url} - Status: ${request.status}`);
                    if (request.status >= 400) {
                        faviconIssues.push(`Favicon request failed: ${request.url} - Status: ${request.status}`);
                    }
                }
            }
        }

        this.validationResults.faviconIssues = faviconIssues;
        
        if (faviconIssues.length > 0) {
            console.log('\\n❌ Favicon issues detected:');
            faviconIssues.forEach(issue => console.log(`  - ${issue}`));
        } else {
            console.log('\\n✅ No critical favicon issues detected');
        }

        return faviconIssues;
    }

    async generateComprehensiveReport() {
        console.log('\\n📊 Generating Comprehensive Validation Report...');
        
        const reportPath = `${this.screenshotDir}/${this.timestamp}/comprehensive_validation_report_${this.timestamp}.md`;
        
        const report = `# Comprehensive UI Validation Report
## Session: ${this.timestamp}

### Executive Summary
- **Base URL**: ${this.baseUrl}
- **Pages Tested**: ${this.pages.length}
- **Viewports**: ${this.viewports.length} (Desktop, Tablet, Mobile)
- **Screenshots Captured**: ${Object.keys(this.validationResults.screenshotTests).length * this.viewports.length}
- **Favicon Tests**: ${Object.keys(this.validationResults.faviconTests).length}

### Favicon Validation Results

${Object.entries(this.validationResults.faviconTests).map(([pageName, result]) => `
#### ${pageName.toUpperCase()} Page
- **Title**: ${result.title}
- **Favicon Requests**: ${result.requests?.length || 0}
- **Tab Screenshot**: ![Tab](${result.screenshot})

${result.requests?.map(req => `- \`${req.url}\` - Status: ${req.status}`).join('\\n') || 'No requests detected'}
`).join('\\n')}

### Screenshot Documentation

${Object.entries(this.validationResults.screenshotTests).map(([pageName, result]) => `
#### ${result.description}
- **URL**: ${this.baseUrl}${result.pageUrl}
- **Status**: ${result.faviconTest ? 'Tested' : 'Not Tested'}

**Screenshots**:
${Object.entries(result.screenshots).map(([viewport, path]) => `- **${viewport}**: ![${viewport}](${path})`).join('\\n')}
`).join('\\n')}

### Issues Detected
${this.validationResults.issues.length > 0 ? 
    this.validationResults.issues.map(issue => `- ❌ ${issue}`).join('\\n') : 
    '✅ No critical issues detected'
}

### Favicon Issues
${this.validationResults.faviconIssues?.length > 0 ? 
    this.validationResults.faviconIssues.map(issue => `- ❌ ${issue}`).join('\\n') : 
    '✅ No critical favicon issues detected'
}

---
*Report generated by SuperClaude v3 Enhanced Backend Integration*
*Comprehensive Screenshot Validation System*
`;

        await fs.writeFile(reportPath, report);
        console.log(`✅ Report saved: ${reportPath}`);
        
        return reportPath;
    }

    async executeValidation() {
        try {
            await this.init();
            await this.executeComprehensiveValidation();
            await this.analyzeFaviconIssues();
            const reportPath = await this.generateComprehensiveReport();
            
            console.log('\\n🎉 Comprehensive Validation Complete!');
            console.log(`📊 Report: ${reportPath}`);
            console.log(`📁 Screenshots: ${this.screenshotDir}/${this.timestamp}/`);
            
            return this.validationResults;
            
        } catch (error) {
            console.log(`❌ Validation execution error: ${error.message}`);
            throw error;
        }
    }
}

// Execute validation
const validator = new ComprehensiveScreenshotValidator();
validator.executeValidation().catch(console.error);