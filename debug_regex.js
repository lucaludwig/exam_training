const fs = require('fs');

function debugFile(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    console.log(`--- Debugging ${filePath} ---`);
    console.log(`Total length: ${content.length}`);
    
    // Check V12.65 pattern
    const match1 = content.match(/QUESTION NO:\s*\d+/);
    console.log(`Match /QUESTION NO:\s*\d+/: ${match1 ? match1[0] : 'None'}`);

    // Check V12.35 pattern
    const match2 = content.match(/NO\.\d+/);
    console.log(`Match /NO\.\d+/: ${match2 ? match2[0] : 'None'}`);
    
    // Check variation
    const match3 = content.match(/NO\.\s*\d+/);
    console.log(`Match /NO\.\s*\d+/: ${match3 ? match3[0] : 'None'}`);

    // Peek at first 500 chars cleaned
    let clean = content.replace(/IT Certification Guaranteed, The Easy Way!/g, '');
    clean = clean.replace(/Page \d+/g, '');
    clean = clean.replace(/Exam\s*:\s*MLA-C01[\s\S]*?Version\s*:\s*V\d+\.\d+/g, '');
    
    console.log("Start of cleaned text:");
    console.log(clean.substring(0, 500));
}

debugFile('MLA-C01 V12.35.txt');
debugFile('MLA-C01 V12.65.txt');

