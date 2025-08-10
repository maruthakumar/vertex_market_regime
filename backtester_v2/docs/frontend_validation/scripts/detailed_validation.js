#!/usr/bin/env node

/**
 * Detailed HTML Structure Validation
 * Deep analysis of the HTML structure to understand what's being rendered
 */

const { chromium } = require('playwright');

async function detailedValidation() {
    console.log('🔍 Starting detailed HTML validation...');
    
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    
    // Enable console logging to see any errors
    page.on('console', msg => {
        console.log(`🖥️  Console ${msg.type()}: ${msg.text()}`);
    });
    
    const testUrl = 'http://173.208.247.17:3001/settings';
    console.log(`\n📄 Testing URL: ${testUrl}`);
    
    try {
        console.log('⏳ Navigating to page...');
        await page.goto(testUrl, { waitUntil: 'networkidle', timeout: 15000 });
        
        console.log('✅ Page loaded successfully');
        
        // Get page title
        const title = await page.title();
        console.log(`📄 Page title: ${title}`);
        
        // Check if it's a Next.js page by looking for Next.js indicators
        const nextData = await page.$('#__next');
        const nextScript = await page.$('script[src*="/_next/"]');
        console.log(`🔧 Next.js indicators - __next div: ${nextData ? 'Found' : 'Missing'}, Next script: ${nextScript ? 'Found' : 'Missing'}`);
        
        // Get the full HTML structure
        const fullHTML = await page.content();
        console.log(`📏 Total HTML length: ${fullHTML.length} characters`);
        
        // Look for semantic elements
        const semantic = {
            header: await page.$$('header'),
            nav: await page.$$('nav'),
            main: await page.$$('main'),
            footer: await page.$$('footer'),
            form: await page.$$('form'),
            input: await page.$$('input'),
            section: await page.$$('section'),
            article: await page.$$('article')
        };
        
        console.log('\n🏗️  Semantic HTML Elements Found:');
        for (const [element, elements] of Object.entries(semantic)) {
            console.log(`   ${element}: ${elements.length} elements`);
            
            // Get some details about the first element if it exists
            if (elements.length > 0) {
                try {
                    const firstElement = elements[0];
                    const tagName = await firstElement.tagName();
                    const className = await firstElement.getAttribute('class');
                    const textPreview = await firstElement.textContent();
                    const preview = textPreview ? textPreview.substring(0, 50).replace(/\s+/g, ' ') : '';
                    console.log(`      First ${tagName}: class="${className || ''}", text="${preview}..."`);
                } catch (e) {
                    console.log(`      (Error getting details: ${e.message})`);
                }
            }
        }
        
        // Check body content structure
        const bodyHTML = await page.innerHTML('body');
        const bodyPreview = bodyHTML.substring(0, 500).replace(/\s+/g, ' ');
        console.log(`\n🏠 Body HTML Preview (first 500 chars):\n${bodyPreview}...`);
        
        // Check if there are any error indicators
        const errorIndicators = await page.$$('.error, [role="alert"], .alert-danger');
        if (errorIndicators.length > 0) {
            console.log(`\n❌ Found ${errorIndicators.length} error indicators on page`);
            for (let i = 0; i < Math.min(errorIndicators.length, 3); i++) {
                const errorText = await errorIndicators[i].textContent();
                console.log(`   Error ${i + 1}: ${errorText?.substring(0, 100)}...`);
            }
        }
        
        // Check network requests
        console.log('\n🌐 Waiting for any additional network requests...');
        await page.waitForTimeout(2000);
        
        console.log('✅ Detailed analysis completed');
        
    } catch (error) {
        console.log(`❌ Error during analysis: ${error.message}`);
        
        // Try to get any content that did load
        try {
            const partialContent = await page.content();
            console.log(`📄 Partial content length: ${partialContent.length}`);
            if (partialContent.length < 1000) {
                console.log(`📄 Partial content:\n${partialContent}`);
            }
        } catch (e) {
            console.log('❌ Could not get partial content');
        }
    }
    
    await browser.close();
    console.log('\n🎉 Detailed validation completed!');
}

// Run the validation
detailedValidation().catch(console.error);