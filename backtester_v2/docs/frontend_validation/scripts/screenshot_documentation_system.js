#!/usr/bin/env node
/**
 * Enterprise GPU Backtester - Screenshot Documentation System
 * 
 * Advanced screenshot management and documentation system implementing
 * SuperClaude v3 Enhanced Backend Integration methodology with comprehensive
 * metadata tracking, automated categorization, and evidence collection.
 * 
 * Phase 4: Documentation & Quality Assurance
 * Component: Screenshot Documentation System with Metadata
 */

const fs = require('fs').promises;
const path = require('path');
const sharp = require('sharp');
const { createHash } = require('crypto');
const { exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

class ScreenshotDocumentationSystem {
    constructor(config = {}) {
        this.config = {
            // Base paths
            screenshotDir: config.screenshotDir || '/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/docs/frontend_validation/screenshots',
            docsDir: config.docsDir || '/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/docs/frontend_validation',
            
            // Documentation structure
            categories: config.categories || [
                'development',
                'production', 
                'comparisons',
                'iterations',
                'before-after',
                'issues',
                'fixes',
                'evidence'
            ],
            
            // Metadata configuration
            generateThumbnails: config.generateThumbnails !== false,
            thumbnailSizes: config.thumbnailSizes || [150, 300, 600],
            compressionQuality: config.compressionQuality || 85,
            
            // Analysis settings
            enableImageAnalysis: config.enableImageAnalysis !== false,
            enableOCR: config.enableOCR === true,
            enableHashComparison: config.enableHashComparison !== false,
            
            // Report generation
            generateHTML: config.generateHTML !== false,
            generateJSON: config.generateJSON !== false,
            generateMarkdown: config.generateMarkdown !== false
        };
        
        this.database = {
            screenshots: new Map(),
            sessions: new Map(),
            comparisons: new Map(),
            evidence: new Map()
        };
        
        this.analytics = {
            totalScreenshots: 0,
            categoryCounts: {},
            sessionStats: {},
            qualityMetrics: {}
        };
    }
    
    /**
     * Initialize documentation system
     */
    async initialize() {
        console.log('📸 Enterprise GPU Backtester - Screenshot Documentation System');
        console.log('=' * 60);
        console.log('SuperClaude v3 Enhanced Backend Integration');
        console.log('Phase 4: Documentation & Quality Assurance');
        console.log('Component: Screenshot Documentation System with Metadata');
        console.log('=' * 60);
        
        // Create directory structure
        await this.createDirectoryStructure();
        
        // Load existing database if available
        await this.loadDatabase();
        
        console.log('📸 Screenshot documentation system initialized');
        console.log(`📁 Base directory: ${this.config.screenshotDir}`);
        console.log(`🗂️ Categories: ${this.config.categories.join(', ')}`);
        console.log(`🔍 Image analysis: ${this.config.enableImageAnalysis ? 'enabled' : 'disabled'}`);
    }
    
    /**
     * Create comprehensive directory structure
     */
    async createDirectoryStructure() {
        const baseDir = this.config.screenshotDir;
        
        // Create main directories
        for (const category of this.config.categories) {
            await this.createDirectory(path.join(baseDir, category));
            await this.createDirectory(path.join(baseDir, category, 'thumbnails'));
            await this.createDirectory(path.join(baseDir, category, 'metadata'));
        }
        
        // Create documentation directories
        await this.createDirectory(path.join(this.config.docsDir, 'reports'));
        await this.createDirectory(path.join(this.config.docsDir, 'evidence'));
        await this.createDirectory(path.join(this.config.docsDir, 'analytics'));
        await this.createDirectory(path.join(this.config.docsDir, 'sessions'));
    }
    
    /**
     * Create directory if it doesn't exist
     */
    async createDirectory(dirPath) {
        try {
            await fs.mkdir(dirPath, { recursive: true });
        } catch (error) {
            if (error.code !== 'EEXIST') {
                throw error;
            }
        }
    }
    
    /**
     * Process and document a screenshot
     */
    async processScreenshot(screenshotPath, metadata = {}) {
        try {
            // Generate unique ID for screenshot
            const screenshotId = await this.generateScreenshotId(screenshotPath);
            
            console.log(`📸 Processing screenshot: ${path.basename(screenshotPath)}`);
            
            // Extract basic file information
            const stats = await fs.stat(screenshotPath);
            const imageBuffer = await fs.readFile(screenshotPath);
            
            // Analyze image properties
            const imageAnalysis = await this.analyzeImage(imageBuffer);
            
            // Generate thumbnails
            const thumbnails = await this.generateThumbnails(screenshotPath, screenshotId);
            
            // Create comprehensive metadata
            const comprehensiveMetadata = {
                id: screenshotId,
                filename: path.basename(screenshotPath),
                originalPath: screenshotPath,
                timestamp: new Date().toISOString(),
                size: stats.size,
                created: stats.birthtime,
                modified: stats.mtime,
                
                // User-provided metadata
                ...metadata,
                
                // Image analysis
                image: imageAnalysis,
                thumbnails: thumbnails,
                
                // Hash for comparison
                hash: this.config.enableHashComparison ? createHash('sha256').update(imageBuffer).digest('hex') : null,
                
                // Quality metrics
                quality: await this.assessImageQuality(imageBuffer),
                
                // OCR data (if enabled)
                ocr: this.config.enableOCR ? await this.performOCR(imageBuffer) : null
            };
            
            // Store in database
            this.database.screenshots.set(screenshotId, comprehensiveMetadata);
            
            // Update analytics
            this.updateAnalytics(comprehensiveMetadata);
            
            // Save metadata file
            await this.saveMetadataFile(screenshotId, comprehensiveMetadata);
            
            console.log(`   ✅ Processed successfully - ID: ${screenshotId}`);
            
            return {
                id: screenshotId,
                metadata: comprehensiveMetadata,
                success: true
            };
            
        } catch (error) {
            console.error(`❌ Failed to process screenshot: ${error.message}`);
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    /**
     * Generate unique screenshot ID
     */
    async generateScreenshotId(screenshotPath) {
        const content = await fs.readFile(screenshotPath);
        const hash = createHash('sha256').update(content).digest('hex');
        const timestamp = Date.now();
        return `${timestamp}_${hash.substring(0, 8)}`;
    }
    
    /**
     * Analyze image properties using Sharp
     */
    async analyzeImage(imageBuffer) {
        try {
            const image = sharp(imageBuffer);
            const metadata = await image.metadata();
            const stats = await image.stats();
            
            return {
                width: metadata.width,
                height: metadata.height,
                format: metadata.format,
                colorSpace: metadata.space,
                channels: metadata.channels,
                density: metadata.density,
                hasAlpha: metadata.hasAlpha,
                isAnimated: metadata.pages > 1,
                
                // Statistical analysis
                stats: {
                    channels: stats.channels.map(channel => ({
                        min: channel.min,
                        max: channel.max,
                        sum: channel.sum,
                        squaredSum: channel.squaredSum,
                        mean: channel.mean,
                        stdev: channel.stdev
                    })),
                    entropy: stats.entropy,
                    sharpness: stats.sharpness
                },
                
                // Calculated metrics
                aspectRatio: metadata.width / metadata.height,
                megapixels: (metadata.width * metadata.height) / 1000000
            };
            
        } catch (error) {
            console.warn(`⚠️ Image analysis failed: ${error.message}`);
            return {
                error: error.message
            };
        }
    }
    
    /**
     * Generate thumbnails at different sizes
     */
    async generateThumbnails(originalPath, screenshotId) {
        const thumbnails = [];
        
        try {
            for (const size of this.config.thumbnailSizes) {
                const thumbnailFilename = `${screenshotId}_${size}.jpg`;
                const thumbnailPath = path.join(
                    this.config.screenshotDir, 
                    'thumbnails', 
                    thumbnailFilename
                );
                
                await sharp(originalPath)
                    .resize(size, null, { 
                        withoutEnlargement: true,
                        fit: 'inside'
                    })
                    .jpeg({ 
                        quality: this.config.compressionQuality,
                        progressive: true
                    })
                    .toFile(thumbnailPath);
                
                thumbnails.push({
                    size: size,
                    filename: thumbnailFilename,
                    path: thumbnailPath
                });
            }
            
            console.log(`   📐 Generated ${thumbnails.length} thumbnails`);
            
        } catch (error) {
            console.warn(`⚠️ Thumbnail generation failed: ${error.message}`);
        }
        
        return thumbnails;
    }
    
    /**
     * Assess image quality metrics
     */
    async assessImageQuality(imageBuffer) {
        try {
            const image = sharp(imageBuffer);
            const stats = await image.stats();
            
            // Calculate quality score based on various factors
            const sharpnessScore = Math.min((stats.sharpness || 0) / 100, 1);
            const contrastScore = this.calculateContrastScore(stats.channels);
            const brightnessScore = this.calculateBrightnessScore(stats.channels);
            
            const overallScore = (sharpnessScore * 0.4 + contrastScore * 0.3 + brightnessScore * 0.3) * 100;
            
            return {
                sharpness: sharpnessScore * 100,
                contrast: contrastScore * 100,
                brightness: brightnessScore * 100,
                overall: overallScore,
                grade: this.gradeImageQuality(overallScore)
            };
            
        } catch (error) {
            return {
                error: error.message,
                overall: 0,
                grade: 'Unknown'
            };
        }
    }
    
    /**
     * Calculate contrast score from channel statistics
     */
    calculateContrastScore(channels) {
        if (!channels || channels.length === 0) return 0;
        
        const avgContrast = channels.reduce((sum, channel) => {
            const range = channel.max - channel.min;
            return sum + (range / 255);
        }, 0) / channels.length;
        
        return Math.min(avgContrast, 1);
    }
    
    /**
     * Calculate brightness score from channel statistics
     */
    calculateBrightnessScore(channels) {
        if (!channels || channels.length === 0) return 0;
        
        const avgBrightness = channels.reduce((sum, channel) => {
            return sum + (channel.mean / 255);
        }, 0) / channels.length;
        
        // Optimal brightness is around 0.5 (middle gray)
        const deviation = Math.abs(avgBrightness - 0.5);
        return Math.max(0, 1 - (deviation * 2));
    }
    
    /**
     * Grade image quality based on overall score
     */
    gradeImageQuality(score) {
        if (score >= 90) return 'Excellent';
        if (score >= 75) return 'Good';
        if (score >= 60) return 'Fair';
        if (score >= 40) return 'Poor';
        return 'Very Poor';
    }
    
    /**
     * Perform OCR on image (placeholder implementation)
     */
    async performOCR(imageBuffer) {
        try {
            // This would integrate with Tesseract.js or similar OCR library
            // For now, return placeholder structure
            return {
                text: '',
                confidence: 0,
                words: [],
                lines: [],
                paragraphs: [],
                enabled: false,
                note: 'OCR implementation placeholder'
            };
        } catch (error) {
            return {
                error: error.message,
                enabled: false
            };
        }
    }
    
    /**
     * Save metadata to JSON file
     */
    async saveMetadataFile(screenshotId, metadata) {
        const metadataPath = path.join(
            this.config.screenshotDir,
            'metadata',
            `${screenshotId}_metadata.json`
        );
        
        await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2));
    }
    
    /**
     * Update analytics with new screenshot data
     */
    updateAnalytics(metadata) {
        this.analytics.totalScreenshots++;
        
        // Category counting
        const category = metadata.category || 'uncategorized';
        this.analytics.categoryCounts[category] = (this.analytics.categoryCounts[category] || 0) + 1;
        
        // Quality metrics aggregation
        if (metadata.quality) {
            if (!this.analytics.qualityMetrics.overall) {
                this.analytics.qualityMetrics.overall = [];
            }
            this.analytics.qualityMetrics.overall.push(metadata.quality.overall);
        }
    }
    
    /**
     * Create session documentation
     */
    async createSessionDocumentation(sessionId, sessionData) {
        console.log(`📝 Creating session documentation: ${sessionId}`);
        
        const sessionDoc = {
            id: sessionId,
            startTime: sessionData.startTime || new Date(),
            endTime: sessionData.endTime || new Date(),
            type: sessionData.type || 'validation',
            
            // Screenshots in session
            screenshots: sessionData.screenshots || [],
            
            // Session metrics
            metrics: {
                totalScreenshots: sessionData.screenshots?.length || 0,
                successRate: sessionData.successRate || 0,
                averageQuality: this.calculateAverageQuality(sessionData.screenshots || []),
                issues: sessionData.issues || []
            },
            
            // Comparison results
            comparisons: sessionData.comparisons || [],
            
            // Evidence collection
            evidence: sessionData.evidence || [],
            
            // Generated reports
            reports: []
        };
        
        // Store in database
        this.database.sessions.set(sessionId, sessionDoc);
        
        // Generate session report
        const sessionReport = await this.generateSessionReport(sessionDoc);
        sessionDoc.reports.push(sessionReport);
        
        console.log(`   ✅ Session documentation created`);
        
        return sessionDoc;
    }
    
    /**
     * Calculate average quality for screenshots
     */
    calculateAverageQuality(screenshots) {
        if (screenshots.length === 0) return 0;
        
        const totalQuality = screenshots.reduce((sum, screenshot) => {
            const metadata = this.database.screenshots.get(screenshot.id);
            return sum + (metadata?.quality?.overall || 0);
        }, 0);
        
        return totalQuality / screenshots.length;
    }
    
    /**
     * Generate comprehensive session report
     */
    async generateSessionReport(sessionDoc) {
        const reportData = {
            metadata: {
                generatedAt: new Date().toISOString(),
                sessionId: sessionDoc.id,
                reportType: 'session_summary'
            },
            
            session: {
                id: sessionDoc.id,
                duration: sessionDoc.endTime - sessionDoc.startTime,
                type: sessionDoc.type,
                screenshots: sessionDoc.metrics.totalScreenshots,
                successRate: sessionDoc.metrics.successRate,
                averageQuality: sessionDoc.metrics.averageQuality
            },
            
            screenshots: sessionDoc.screenshots.map(screenshot => {
                const metadata = this.database.screenshots.get(screenshot.id);
                return {
                    id: screenshot.id,
                    filename: metadata?.filename,
                    quality: metadata?.quality,
                    image: {
                        width: metadata?.image?.width,
                        height: metadata?.image?.height,
                        format: metadata?.image?.format
                    }
                };
            }),
            
            issues: sessionDoc.metrics.issues,
            comparisons: sessionDoc.comparisons,
            evidence: sessionDoc.evidence
        };
        
        // Save report files
        const reports = {};
        
        if (this.config.generateJSON) {
            const jsonPath = path.join(this.config.docsDir, 'sessions', `${sessionDoc.id}_report.json`);
            await fs.writeFile(jsonPath, JSON.stringify(reportData, null, 2));
            reports.json = jsonPath;
        }
        
        if (this.config.generateHTML) {
            const htmlPath = path.join(this.config.docsDir, 'sessions', `${sessionDoc.id}_report.html`);
            const htmlContent = await this.generateHTMLReport(reportData);
            await fs.writeFile(htmlPath, htmlContent);
            reports.html = htmlPath;
        }
        
        if (this.config.generateMarkdown) {
            const mdPath = path.join(this.config.docsDir, 'sessions', `${sessionDoc.id}_report.md`);
            const mdContent = await this.generateMarkdownReport(reportData);
            await fs.writeFile(mdPath, mdContent);
            reports.markdown = mdPath;
        }
        
        console.log(`   📋 Generated session reports: ${Object.keys(reports).join(', ')}`);
        
        return {
            paths: reports,
            data: reportData
        };
    }
    
    /**
     * Generate HTML report
     */
    async generateHTMLReport(reportData) {
        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Session Report - ${reportData.session.id}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2rem; }
        .header { border-bottom: 2px solid #3b82f6; padding-bottom: 1rem; margin-bottom: 2rem; }
        .metric { display: inline-block; margin: 0.5rem 1rem 0.5rem 0; padding: 0.5rem 1rem; background: #f3f4f6; border-radius: 0.5rem; }
        .screenshot { margin: 1rem 0; padding: 1rem; border: 1px solid #e5e7eb; border-radius: 0.5rem; }
        .quality-excellent { color: #10b981; }
        .quality-good { color: #3b82f6; }
        .quality-fair { color: #f59e0b; }
        .quality-poor { color: #ef4444; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Enterprise GPU Backtester - Session Report</h1>
        <p>Session: ${reportData.session.id} | Generated: ${reportData.metadata.generatedAt}</p>
    </div>
    
    <section>
        <h2>Session Summary</h2>
        <div class="metric">Screenshots: ${reportData.session.screenshots}</div>
        <div class="metric">Success Rate: ${(reportData.session.successRate * 100).toFixed(1)}%</div>
        <div class="metric">Avg Quality: ${reportData.session.averageQuality.toFixed(1)}</div>
        <div class="metric">Duration: ${(reportData.session.duration / 1000).toFixed(1)}s</div>
    </section>
    
    <section>
        <h2>Screenshots (${reportData.screenshots.length})</h2>
        ${reportData.screenshots.map(screenshot => `
            <div class="screenshot">
                <h3>${screenshot.filename}</h3>
                <p>Quality: <span class="quality-${screenshot.quality?.grade?.toLowerCase()}">${screenshot.quality?.grade} (${screenshot.quality?.overall?.toFixed(1)})</span></p>
                <p>Dimensions: ${screenshot.image?.width}x${screenshot.image?.height} ${screenshot.image?.format}</p>
            </div>
        `).join('')}
    </section>
    
    ${reportData.issues.length > 0 ? `
    <section>
        <h2>Issues (${reportData.issues.length})</h2>
        ${reportData.issues.map(issue => `
            <div class="screenshot">
                <h3>${issue.severity} - ${issue.category}</h3>
                <p>${issue.description}</p>
            </div>
        `).join('')}
    </section>
    ` : ''}
    
    <footer style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 0.875rem;">
        <p>Generated by Enterprise GPU Backtester UI Validation System - SuperClaude v3 Enhanced Backend Integration</p>
    </footer>
</body>
</html>`;
    }
    
    /**
     * Generate Markdown report
     */
    async generateMarkdownReport(reportData) {
        return `# Enterprise GPU Backtester - Session Report

**Session**: ${reportData.session.id}  
**Generated**: ${reportData.metadata.generatedAt}  
**Type**: ${reportData.session.type}

## Session Summary

- **Screenshots**: ${reportData.session.screenshots}
- **Success Rate**: ${(reportData.session.successRate * 100).toFixed(1)}%
- **Average Quality**: ${reportData.session.averageQuality.toFixed(1)}
- **Duration**: ${(reportData.session.duration / 1000).toFixed(1)}s

## Screenshots (${reportData.screenshots.length})

${reportData.screenshots.map(screenshot => `
### ${screenshot.filename}

- **Quality**: ${screenshot.quality?.grade} (${screenshot.quality?.overall?.toFixed(1)})
- **Dimensions**: ${screenshot.image?.width}x${screenshot.image?.height} ${screenshot.image?.format}
- **ID**: \`${screenshot.id}\`
`).join('')}

${reportData.issues.length > 0 ? `
## Issues (${reportData.issues.length})

${reportData.issues.map(issue => `
### ${issue.severity} - ${issue.category}

${issue.description}
`).join('')}
` : ''}

---

*Generated by Enterprise GPU Backtester UI Validation System - SuperClaude v3 Enhanced Backend Integration*`;
    }
    
    /**
     * Load existing database from files
     */
    async loadDatabase() {
        try {
            const dbPath = path.join(this.config.docsDir, 'screenshot_database.json');
            const dbContent = await fs.readFile(dbPath, 'utf8');
            const dbData = JSON.parse(dbContent);
            
            // Restore Maps from JSON
            if (dbData.screenshots) {
                this.database.screenshots = new Map(Object.entries(dbData.screenshots));
            }
            if (dbData.sessions) {
                this.database.sessions = new Map(Object.entries(dbData.sessions));
            }
            
            console.log(`📂 Loaded existing database (${this.database.screenshots.size} screenshots)`);
            
        } catch (error) {
            console.log('📂 No existing database found, starting fresh');
        }
    }
    
    /**
     * Save database to file
     */
    async saveDatabase() {
        const dbData = {
            screenshots: Object.fromEntries(this.database.screenshots),
            sessions: Object.fromEntries(this.database.sessions),
            comparisons: Object.fromEntries(this.database.comparisons),
            evidence: Object.fromEntries(this.database.evidence),
            analytics: this.analytics,
            lastUpdated: new Date().toISOString()
        };
        
        const dbPath = path.join(this.config.docsDir, 'screenshot_database.json');
        await fs.writeFile(dbPath, JSON.stringify(dbData, null, 2));
        
        console.log(`💾 Database saved (${this.database.screenshots.size} screenshots)`);
    }
    
    /**
     * Generate comprehensive analytics report
     */
    async generateAnalyticsReport() {
        const analytics = {
            metadata: {
                generatedAt: new Date().toISOString(),
                reportType: 'analytics_summary'
            },
            
            overview: {
                totalScreenshots: this.analytics.totalScreenshots,
                totalSessions: this.database.sessions.size,
                categoryCounts: this.analytics.categoryCounts
            },
            
            quality: {
                averageQuality: this.analytics.qualityMetrics.overall ? 
                    this.analytics.qualityMetrics.overall.reduce((a, b) => a + b, 0) / this.analytics.qualityMetrics.overall.length : 0,
                qualityDistribution: this.calculateQualityDistribution()
            },
            
            sessions: Array.from(this.database.sessions.values()).map(session => ({
                id: session.id,
                type: session.type,
                screenshots: session.metrics.totalScreenshots,
                successRate: session.metrics.successRate,
                averageQuality: session.metrics.averageQuality
            }))
        };
        
        const reportPath = path.join(this.config.docsDir, 'analytics', `analytics_report_${Date.now()}.json`);
        await fs.writeFile(reportPath, JSON.stringify(analytics, null, 2));
        
        console.log(`📊 Analytics report generated: ${reportPath}`);
        
        return analytics;
    }
    
    /**
     * Calculate quality distribution
     */
    calculateQualityDistribution() {
        if (!this.analytics.qualityMetrics.overall) return {};
        
        const distribution = { excellent: 0, good: 0, fair: 0, poor: 0, veryPoor: 0 };
        
        for (const quality of this.analytics.qualityMetrics.overall) {
            if (quality >= 90) distribution.excellent++;
            else if (quality >= 75) distribution.good++;
            else if (quality >= 60) distribution.fair++;
            else if (quality >= 40) distribution.poor++;
            else distribution.veryPoor++;
        }
        
        return distribution;
    }
}

/**
 * Main execution function
 */
async function main() {
    const docSystem = new ScreenshotDocumentationSystem();
    
    try {
        await docSystem.initialize();
        
        // Example usage - process screenshots from validation system
        const screenshotDir = '/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/docs/frontend_validation/screenshots';
        
        console.log('🔍 Scanning for screenshots to process...');
        
        // This would integrate with the validation system
        console.log('📸 Screenshot documentation system ready for integration');
        console.log('Use docSystem.processScreenshot() to process individual screenshots');
        console.log('Use docSystem.createSessionDocumentation() for session reports');
        
        await docSystem.generateAnalyticsReport();
        await docSystem.saveDatabase();
        
    } catch (error) {
        console.error(`❌ Documentation system failed: ${error.message}`);
        console.error(error.stack);
        process.exit(1);
    }
}

// Execute if run directly
if (require.main === module) {
    main();
}

module.exports = { ScreenshotDocumentationSystem };