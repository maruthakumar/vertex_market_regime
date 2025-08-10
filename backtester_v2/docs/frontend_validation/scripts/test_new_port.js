#!/usr/bin/env node

/**
 * Test New Port Validation
 * Check if the new Next.js application on port 3001 has semantic HTML elements
 */

const { chromium } = require('playwright');

async function testNewPort() {
    console.log('🔍 Testing Next.js app on port 3001...');
    
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    
    const pagesToTest = [
        { name: 'Backtest', url: 'http://173.208.247.17:3001/backtest' },
        { name: 'Results', url: 'http://173.208.247.17:3001/results' },
        { name: 'Settings', url: 'http://173.208.247.17:3001/settings' }
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
            
            // Get first few elements of the body to understand structure
            const bodyText = await page.textContent('body');
            if (bodyText) {
                const firstWords = bodyText.replace(/\s+/g, ' ').trim().substring(0, 100);
                console.log(`   📖 First content: ${firstWords}...`);
            }
            
        } catch (error) {
            console.log(`   ❌ Error loading ${testPage.name}: ${error.message}`);
        }
    }
    
    await browser.close();
    console.log('\n🎉 Port 3001 validation completed!');
}

// Run the validation
testNewPort().catch(console.error);