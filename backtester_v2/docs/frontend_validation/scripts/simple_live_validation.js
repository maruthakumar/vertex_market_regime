#!/usr/bin/env node
/**
 * Simple Live UI Validation for Enterprise GPU Backtester
 * 
 * Performs real UI validation against the live application
 * at http://173.208.247.17:3000/ to identify actual issues
 */

const fs = require('fs').promises;
const path = require('path');
const { chromium } = require('playwright');

class SimpleLiveValidator {
    constructor() {
        this.targetUrl = 'http://173.208.247.17:3000/';
        this.resultsDir = '/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/docs/frontend_validation/live-validation-results';
        this.screenshotsDir = path.join(this.resultsDir, 'screenshots');
        this.issuesDir = path.join(this.resultsDir, 'issues');
        
        this.issues = [];
        this.screenshots = [];
        
        this.testPages = [
            { name: 'Home', path: '/', critical: true },
            { name: 'Dashboard', path: '/dashboard', critical: true },
            { name: 'Backtest', path: '/backtest', critical: true },
            { name: 'Results', path: '/results', critical: false },
            { name: 'Settings', path: '/settings', critical: false }
        ];
    }
    
    async initialize() {
        console.log('🔴 Simple Live UI Validation - Enterprise GPU Backtester');
        console.log('=' * 60);
        console.log(`🎯 Target: ${this.targetUrl}`);
        console.log('=' * 60);
        
        // Create directories
        await this.createDir(this.resultsDir);
        await this.createDir(this.screenshotsDir);
        await this.createDir(this.issuesDir);
        
        console.log('✅ Validation system initialized');
    }
    
    async validateApplication() {
        const startTime = Date.now();
        console.log('🚀 Starting live application validation...\n');
        
        const browser = await chromium.launch({ headless: true });
        const context = await browser.newContext({
            viewport: { width: 1920, height: 1080 }
        });
        
        try {
            for (const testPage of this.testPages) {
                console.log(`📄 Testing: ${testPage.name} (${testPage.path})`);
                
                const page = await context.newPage();
                
                try {
                    // Navigate to page
                    const url = this.targetUrl + testPage.path.slice(1);
                    console.log(`   🌐 Navigating to: ${url}`);
                    
                    const response = await page.goto(url, {
                        waitUntil: 'networkidle',
                        timeout: 30000
                    });
                    
                    if (!response || response.status() >= 400) {
                        this.addIssue({
                            type: 'navigation',
                            severity: testPage.critical ? 'CRITICAL' : 'HIGH',
                            page: testPage.name,
                            description: `Failed to load page: HTTP ${response?.status() || 'timeout'}`,
                            url: url
                        });
                        console.log(`   ❌ Failed to load page: ${response?.status() || 'timeout'}`);
                        continue;
                    }
                    
                    // Wait for page to stabilize
                    await page.waitForTimeout(2000);
                    
                    // Capture screenshot
                    const screenshotPath = await this.captureScreenshot(page, testPage.name);
                    console.log(`   📸 Screenshot: ${path.basename(screenshotPath)}`);
                    
                    // Test page functionality
                    const pageIssues = await this.testPageFunctionality(page, testPage);
                    this.issues.push(...pageIssues);
                    
                    if (pageIssues.length > 0) {
                        console.log(`   🚨 Found ${pageIssues.length} issues`);
                    } else {
                        console.log(`   ✅ No major issues detected`);
                    }
                    
                } catch (error) {
                    this.addIssue({
                        type: 'page-error',
                        severity: 'HIGH',
                        page: testPage.name,
                        description: `Page testing failed: ${error.message}`,
                        error: error.stack
                    });
                    console.log(`   ❌ Testing error: ${error.message}`);
                }
                
                await page.close();
                console.log(''); // Empty line for readability
            }
            
        } finally {
            await browser.close();
        }
        
        const endTime = Date.now();
        const duration = (endTime - startTime) / 1000;
        
        // Generate results
        const results = await this.generateResults(duration);
        
        console.log('🎉 Live validation completed!');
        console.log(`⏱️  Duration: ${duration.toFixed(2)}s`);
        console.log(`🔍 Issues found: ${this.issues.length}`);
        console.log(`📸 Screenshots: ${this.screenshots.length}`);
        
        return results;
    }
    
    async testPageFunctionality(page, testPage) {
        const pageIssues = [];
        
        try {
            // Check for JavaScript errors
            const consoleErrors = [];
            page.on('console', msg => {
                if (msg.type() === 'error') {
                    consoleErrors.push(msg.text());
                }
            });
            
            // Test common UI elements
            const elementTests = [
                { selector: 'nav', name: 'Navigation' },
                { selector: 'header', name: 'Header' },
                { selector: 'main', name: 'Main content' },
                { selector: 'footer', name: 'Footer' },
                { selector: 'button', name: 'Buttons' },
                { selector: 'form', name: 'Forms' },
                { selector: 'input', name: 'Inputs' }
            ];
            
            for (const test of elementTests) {
                try {
                    const element = await page.$(test.selector);
                    if (!element) {
                        pageIssues.push({
                            type: 'missing-element',
                            severity: 'MEDIUM',
                            page: testPage.name,
                            description: `Missing ${test.name} element (${test.selector})`,
                            selector: test.selector
                        });
                    }
                } catch (error) {
                    pageIssues.push({
                        type: 'element-error',
                        severity: 'LOW',
                        page: testPage.name,
                        description: `Error testing ${test.name}: ${error.message}`,
                        selector: test.selector
                    });
                }
            }
            
            // Check for broken images
            const images = await page.$$('img');
            for (let i = 0; i < images.length; i++) {
                try {
                    const src = await images[i].getAttribute('src');
                    const naturalWidth = await images[i].evaluate(img => img.naturalWidth);
                    
                    if (naturalWidth === 0) {
                        pageIssues.push({
                            type: 'broken-image',
                            severity: 'MEDIUM',
                            page: testPage.name,
                            description: `Broken image: ${src || 'unknown source'}`,
                            src: src
                        });
                    }
                } catch (error) {
                    // Ignore image test errors
                }
            }
            
            // Check for layout issues
            const layoutIssues = await this.checkLayoutIssues(page, testPage);
            pageIssues.push(...layoutIssues);
            
            // Check for accessibility issues
            const accessibilityIssues = await this.checkAccessibilityIssues(page, testPage);
            pageIssues.push(...accessibilityIssues);
            
            // Report console errors
            if (consoleErrors.length > 0) {
                pageIssues.push({
                    type: 'console-error',
                    severity: 'MEDIUM',
                    page: testPage.name,
                    description: `JavaScript console errors: ${consoleErrors.length}`,
                    errors: consoleErrors
                });
            }
            
        } catch (error) {
            pageIssues.push({
                type: 'functionality-test-error',
                severity: 'LOW',
                page: testPage.name,
                description: `Functionality testing error: ${error.message}`,
                error: error.stack
            });
        }
        
        return pageIssues;
    }
    
    async checkLayoutIssues(page, testPage) {
        const layoutIssues = [];
        
        try {
            // Check for elements that might be off-screen or overlapping
            const elements = await page.$$('*');
            
            // Sample some elements for performance
            const sampleElements = elements.slice(0, Math.min(50, elements.length));
            
            for (const element of sampleElements) {
                try {
                    const box = await element.boundingBox();
                    if (box) {
                        // Check if element is way off-screen (potential layout issue)
                        if (box.x < -1000 || box.y < -1000) {
                            layoutIssues.push({
                                type: 'layout-positioning',
                                severity: 'LOW',
                                page: testPage.name,
                                description: `Element positioned far off-screen: x=${box.x}, y=${box.y}`,
                                position: { x: box.x, y: box.y }
                            });
                        }
                        
                        // Check for extremely wide or tall elements
                        if (box.width > 5000 || box.height > 5000) {
                            layoutIssues.push({
                                type: 'layout-size',
                                severity: 'MEDIUM',
                                page: testPage.name,
                                description: `Extremely large element: ${box.width}x${box.height}`,
                                size: { width: box.width, height: box.height }
                            });
                        }
                    }
                } catch (error) {
                    // Ignore individual element errors
                }
            }
        } catch (error) {
            // Ignore layout check errors
        }
        
        return layoutIssues;
    }
    
    async checkAccessibilityIssues(page, testPage) {
        const accessibilityIssues = [];
        
        try {
            // Check for missing alt attributes on images
            const imagesWithoutAlt = await page.$$eval('img:not([alt])', imgs => imgs.length);
            if (imagesWithoutAlt > 0) {
                accessibilityIssues.push({
                    type: 'accessibility-alt',
                    severity: 'MEDIUM',
                    page: testPage.name,
                    description: `${imagesWithoutAlt} images missing alt attributes`,
                    count: imagesWithoutAlt
                });
            }
            
            // Check for forms without labels
            const unlabeledInputs = await page.$$eval('input:not([aria-label]):not([aria-labelledby])', inputs => {
                return inputs.filter(input => {
                    const label = input.closest('label') || document.querySelector(`label[for="${input.id}"]`);
                    return !label;
                }).length;
            });
            
            if (unlabeledInputs > 0) {
                accessibilityIssues.push({
                    type: 'accessibility-labels',
                    severity: 'MEDIUM',
                    page: testPage.name,
                    description: `${unlabeledInputs} form inputs missing labels`,
                    count: unlabeledInputs
                });
            }
            
            // Check for insufficient color contrast (basic check)
            const hasColorIssues = await page.evaluate(() => {
                // Basic check for very light gray text on white backgrounds
                const elements = Array.from(document.querySelectorAll('*'));
                return elements.some(el => {
                    const style = window.getComputedStyle(el);
                    const color = style.color;
                    const backgroundColor = style.backgroundColor;
                    
                    // Very basic contrast check - this is simplified
                    return color.includes('rgb(200,') && backgroundColor.includes('rgb(255,');
                });
            });
            
            if (hasColorIssues) {
                accessibilityIssues.push({
                    type: 'accessibility-contrast',
                    severity: 'LOW',
                    page: testPage.name,
                    description: 'Potential color contrast issues detected'
                });
            }
            
        } catch (error) {
            // Ignore accessibility check errors
        }
        
        return accessibilityIssues;
    }
    
    async captureScreenshot(page, pageName) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `${pageName.toLowerCase()}_${timestamp}.png`;
        const screenshotPath = path.join(this.screenshotsDir, filename);
        
        await page.screenshot({ 
            path: screenshotPath, 
            fullPage: true 
        });
        
        this.screenshots.push({
            page: pageName,
            path: screenshotPath,
            timestamp: new Date()
        });
        
        return screenshotPath;
    }
    
    addIssue(issue) {
        this.issues.push({
            ...issue,
            id: `issue_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            timestamp: new Date()
        });
    }
    
    async generateResults(duration) {
        // Categorize issues
        const issuesByType = {};
        const issuesBySeverity = {};
        
        for (const issue of this.issues) {
            // By type
            if (!issuesByType[issue.type]) {
                issuesByType[issue.type] = [];
            }
            issuesByType[issue.type].push(issue);
            
            // By severity
            if (!issuesBySeverity[issue.severity]) {
                issuesBySeverity[issue.severity] = [];
            }
            issuesBySeverity[issue.severity].push(issue);
        }
        
        const results = {
            sessionId: `validation_${Date.now()}`,
            timestamp: new Date(),
            duration: duration,
            targetUrl: this.targetUrl,
            summary: {
                totalIssues: this.issues.length,
                criticalIssues: (issuesBySeverity.CRITICAL || []).length,
                highIssues: (issuesBySeverity.HIGH || []).length,
                mediumIssues: (issuesBySeverity.MEDIUM || []).length,
                lowIssues: (issuesBySeverity.LOW || []).length,
                screenshotsCaptured: this.screenshots.length
            },
            issues: this.issues,
            screenshots: this.screenshots,
            issuesByType,
            issuesBySeverity
        };
        
        // Save results
        const resultsPath = path.join(this.resultsDir, `validation_results_${results.sessionId}.json`);
        await fs.writeFile(resultsPath, JSON.stringify(results, null, 2));
        
        // Generate human-readable report
        const reportPath = path.join(this.resultsDir, `validation_report_${results.sessionId}.md`);
        await this.generateMarkdownReport(results, reportPath);
        
        console.log(`📋 Results saved: ${path.basename(resultsPath)}`);
        console.log(`📄 Report saved: ${path.basename(reportPath)}`);
        
        return results;
    }
    
    async generateMarkdownReport(results, reportPath) {
        const report = `# Live UI Validation Report

## Summary
- **Target Application**: ${results.targetUrl}
- **Validation Date**: ${results.timestamp}
- **Duration**: ${results.duration.toFixed(2)} seconds
- **Total Issues Found**: ${results.summary.totalIssues}

## Issue Breakdown
- 🔴 **Critical**: ${results.summary.criticalIssues}
- 🟠 **High**: ${results.summary.highIssues}
- 🟡 **Medium**: ${results.summary.mediumIssues}
- 🟢 **Low**: ${results.summary.lowIssues}

## Issues by Type
${Object.entries(results.issuesByType).map(([type, issues]) => 
    `- **${type}**: ${issues.length} issues`
).join('\n')}

## Detailed Issues

${results.issues.map(issue => `
### ${issue.severity} - ${issue.type}
- **Page**: ${issue.page}
- **Description**: ${issue.description}
- **Issue ID**: ${issue.id}
- **Timestamp**: ${issue.timestamp}
${issue.selector ? `- **Selector**: ${issue.selector}` : ''}
${issue.url ? `- **URL**: ${issue.url}` : ''}
${issue.errors ? `- **Errors**: ${issue.errors.length} console errors` : ''}
`).join('\n')}

## Screenshots Captured
${results.screenshots.map(screenshot => 
    `- **${screenshot.page}**: ${path.basename(screenshot.path)}`
).join('\n')}

## Next Steps
1. Review critical and high severity issues first
2. Fix navigation and functionality problems
3. Address accessibility concerns
4. Optimize layout and visual issues
5. Re-run validation to confirm fixes

---
Generated by Enterprise GPU Backtester UI Validation System
`;
        
        await fs.writeFile(reportPath, report);
    }
    
    async createDir(dirPath) {
        try {
            await fs.mkdir(dirPath, { recursive: true });
        } catch (error) {
            if (error.code !== 'EEXIST') throw error;
        }
    }
}

async function main() {
    const validator = new SimpleLiveValidator();
    
    try {
        await validator.initialize();
        const results = await validator.validateApplication();
        
        console.log('\n🎯 VALIDATION SUMMARY:');
        console.log(`   Total Issues: ${results.summary.totalIssues}`);
        console.log(`   Critical: ${results.summary.criticalIssues}`);
        console.log(`   High: ${results.summary.highIssues}`);
        console.log(`   Medium: ${results.summary.mediumIssues}`);
        console.log(`   Low: ${results.summary.lowIssues}`);
        console.log(`   Screenshots: ${results.summary.screenshotsCaptured}`);
        
        if (results.summary.totalIssues > 0) {
            console.log('\n📋 Issues found! Check the detailed report for more information.');
        } else {
            console.log('\n✅ No issues detected! Application appears to be working well.');
        }
        
        return results;
        
    } catch (error) {
        console.error(`❌ Validation failed: ${error.message}`);
        console.error(error.stack);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = { SimpleLiveValidator };