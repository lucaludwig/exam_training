const fs = require('fs');

function parseQuestionsJs(filePath) {
    if (!fs.existsSync(filePath)) return [];
    const content = fs.readFileSync(filePath, 'utf8');
    const match = content.match(/const quizData = ([\s\S]*]);/);
    if (match && match[1]) {
        try {
            return JSON.parse(match[1]);
        } catch (e) {
            console.error("Error parsing existing questions.js JSON:", e);
            return [];
        }
    }
    return [];
}

function normalizeText(text) {
    if (!text) return "";
    // Normalize to simple lowercase alphanumeric string for duplicate checking
    return text.toLowerCase().replace(/[^a-z0-9]/g, '');
}

function parseTextFile(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Cleanup headers/footers
    content = content.replace(/IT Certification Guaranteed, The Easy Way!/g, '');
    content = content.replace(/Page \d+/g, '');
    content = content.replace(/\n\s*\d+\s*\n/g, '\n');
    content = content.replace(/Exam\s*:\s*MLA-C01[\s\S]*?Version\s*:\s*V\d+\.\d+/g, '');

    const rawQuestions = content.split(/QUESTION NO:\s*\d+/).slice(1);
    
    const questions = [];

    rawQuestions.forEach(raw => {
        let text = raw.trim();

        // Extract Answer
        const answerMatch = text.match(/Answer:\s*([A-Z,]+)/);
        let answer = answerMatch ? answerMatch[1].trim() : null;

        if (!answer) return; 

        // Extract Explanation
        let explanation = null;
        const explanationMatch = text.match(/Explanation:\s*([\s\S]*?)$/);
        if (explanationMatch) {
            explanation = explanationMatch[1].trim();
            text = text.substring(0, explanationMatch.index).trim();
        }

        // Remove Answer line
        text = text.replace(/Answer:\s*[A-Z,]+.*/, '').trim();

        let questionText = "";
        let options = [];
        
        // Option parsing logic
        const optionMarkers = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.'];
        
        // Find positions of options
        // We look for "A."
        let firstOptionIndex = -1;
        
        // Robust search for "A." followed eventually by "B."
        const matchesA = [...text.matchAll(/(^|\n|\s)A\.\s/g)];
        for (const match of matchesA) {
            const idx = match.index + match[0].indexOf('A');
            // Check if B. follows
            if (text.indexOf('B.', idx) > idx) {
                firstOptionIndex = idx;
                break;
            }
        }

        if (firstOptionIndex !== -1) {
            questionText = text.substring(0, firstOptionIndex).trim();
            const optionsBlock = text.substring(firstOptionIndex).trim();
            
            // Map of label -> start index
            const labelIndices = {};
            
            // Helper to find label position
            const findLabel = (block, label, startFrom) => {
                const regex = new RegExp(`(^|\\n|\\s)${label}\.\s`);
                const match = block.substring(startFrom).match(regex);
                if (match) {
                    return startFrom + match.index + match[0].indexOf(label);
                }
                return -1;
            };

            // Find all present labels in order
            let lastIdx = 0;
            for (const label of optionMarkers) {
                const idx = findLabel(optionsBlock, label, lastIdx);
                if (idx !== -1) {
                    labelIndices[label] = idx;
                    lastIdx = idx; // Optimization: next option must be after this one
                }
            }

            const sortedLabels = Object.keys(labelIndices).sort();
            
            for (let i = 0; i < sortedLabels.length; i++) {
                const label = sortedLabels[i];
                const start = labelIndices[label];
                const nextLabelKey = sortedLabels[i+1];
                const end = nextLabelKey ? labelIndices[nextLabelKey] : optionsBlock.length;
                
                let optText = optionsBlock.substring(start, end).trim();
                // Remove the "A. " prefix
                optText = optText.replace(/^[A-Z]\.\s/, '');
                options.push(`${label}. ${optText}`);
            }

        } else {
             questionText = text;
        }

        if (questionText && options.length > 0) {
            const qObj = {
                question: questionText,
                options: options,
                answer: answer
            };
            if (explanation) {
                qObj.explanation = explanation;
            }
            questions.push(qObj);
        }
    });

    return questions;
}

function mergeAll() {
    // 1. Load current questions (which serves as our base, but we will rebuild duplicate check)
    let finalQuestions = parseQuestionsJs('questions.js');
    console.log(`Starting with: ${finalQuestions.length} questions.`);

    // 2. Identify files to process
    const files = ['MLA-C01 V12.35.txt', 'MLA-C01 V12.65.txt'];

    // Create a Set of normalized questions for deduplication
    const existingSet = new Set();
    finalQuestions.forEach(q => existingSet.add(normalizeText(q.question)));

    let totalAdded = 0;

    files.forEach(file => {
        if (fs.existsSync(file)) {
            console.log(`Processing ${file}...`);
            const newQs = parseTextFile(file);
            console.log(`  Found ${newQs.length} questions.`);
            
            let addedFromFile = 0;
            newQs.forEach(q => {
                const normQ = normalizeText(q.question);
                // Basic length check to avoid garbage
                if (q.question.length < 10) return;

                if (!existingSet.has(normQ)) {
                    finalQuestions.push(q);
                    existingSet.add(normQ);
                    addedFromFile++;
                    totalAdded++;
                }
            });
            console.log(`  Added ${addedFromFile} unique questions.`);
        } else {
            console.log(`File not found: ${file}`);
        }
    });

    console.log(`Total new questions added: ${totalAdded}`);
    console.log(`Final total questions: ${finalQuestions.length}`);

    const outputContent = `const quizData = ${JSON.stringify(finalQuestions, null, 4)};`;
    fs.writeFileSync('questions.js', outputContent);
}

mergeAll();
