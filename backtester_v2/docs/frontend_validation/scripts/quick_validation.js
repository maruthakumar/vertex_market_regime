#!/usr/bin/env node

/**
 * Quick Semantic HTML Element Validation
 * Test specific pages for semantic HTML elements
 */

const { chromium } = require('playwright');

async function quickValidation() {
    console.log('🔍 Quick semantic HTML validation starting...');
    
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    
    const pagesToTest = [
        { name: 'Backtest', url: 'http://173.208.247.17:3000/backtest' },
        { name: 'Results', url: 'http://173.208.247.17:3000/results' },
        { name: 'Settings', url: 'http://173.208.247.17:3000/settings' }
    ];
    
    for (const testPage of pagesToTest) {
        console.log(`\n📄 Testing: ${testPage.name} (${testPage.url})`);
        
        try {
            await page.goto(testPage.url, { waitUntil: 'networkidle', timeout: 10000 });
            
            // Check for semantic HTML elements
            const header = await page.$('header');
            const nav = await page.$('nav');
            const main = await page.$('main');
            const footer = await page.$('footer');
            const form = await page.$('form');
            const input = await page.$('input');
            
            console.log(`   ✅ Header element: ${header ? 'Found' : 'Missing'}`);
            console.log(`   ✅ Navigation element: ${nav ? 'Found' : 'Missing'}`);
            console.log(`   ✅ Main element: ${main ? 'Found' : 'Missing'}`);
            console.log(`   ✅ Footer element: ${footer ? 'Found' : 'Missing'}`);
            console.log(`   ✅ Form element: ${form ? 'Found' : 'Missing'}`);
            console.log(`   ✅ Input element: ${input ? 'Found' : 'Missing'}`);
            
            // Get page content summary
            const title = await page.title();
            console.log(`   📄 Page title: ${title}`);
            
        } catch (error) {
            console.log(`   ❌ Error loading ${testPage.name}: ${error.message}`);
        }
    }
    
    await browser.close();
    console.log('\n🎉 Quick validation completed!');
}

// Run the validation
quickValidation().catch(console.error);