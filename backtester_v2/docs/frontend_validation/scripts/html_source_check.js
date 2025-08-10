#!/usr/bin/env node

/**
 * HTML Source Check
 * Directly examine the HTML source to see if semantic elements are present
 */

const { chromium } = require('playwright');

async function checkHTMLSource() {
    console.log('🔍 Checking HTML source for semantic elements...');
    
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    
    const testUrl = 'http://173.208.247.17:3000/backtest';
    
    console.log(`\n📄 Testing URL: ${testUrl}`);
    
    try {
        await page.goto(testUrl, { waitUntil: 'networkidle', timeout: 10000 });
        
        // Get the full HTML source
        const htmlSource = await page.content();
        
        // Search for semantic elements in the HTML source
        const hasHeader = htmlSource.includes('<header');
        const hasNav = htmlSource.includes('<nav');
        const hasMain = htmlSource.includes('<main');
        const hasFooter = htmlSource.includes('<footer');
        const hasForm = htmlSource.includes('<form');
        const hasInput = htmlSource.includes('<input');
        
        console.log('\n🔍 Semantic elements in HTML source:');
        console.log(`   Header (<header): ${hasHeader ? '✅ Found' : '❌ Missing'}`);
        console.log(`   Navigation (<nav): ${hasNav ? '✅ Found' : '❌ Missing'}`);
        console.log(`   Main (<main): ${hasMain ? '✅ Found' : '❌ Missing'}`);
        console.log(`   Footer (<footer): ${hasFooter ? '✅ Found' : '❌ Missing'}`);
        console.log(`   Form (<form): ${hasForm ? '✅ Found' : '❌ Missing'}`);
        console.log(`   Input (<input): ${hasInput ? '✅ Found' : '❌ Missing'}`);
        
        // Extract a sample of the body content
        const bodyMatch = htmlSource.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
        if (bodyMatch) {
            const bodyContent = bodyMatch[1];
            const firstKB = bodyContent.substring(0, 1000);
            console.log('\n📄 First 1KB of body content:');
            console.log('---');
            console.log(firstKB.replace(/\n\s*/g, ' ').trim());
            console.log('---');
        }
        
    } catch (error) {
        console.log(`   ❌ Error loading page: ${error.message}`);
    }
    
    await browser.close();
    console.log('\n🎉 HTML source check completed!');
}

// Run the check
checkHTMLSource().catch(console.error);