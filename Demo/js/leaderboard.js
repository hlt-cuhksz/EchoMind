// EchoMind Leaderboard JavaScript

// Merge data for models that appear multiple times
function mergeLeaderboardData(data) {
    const mergedData = {};
    
    data.leaderboardData.forEach(entry => {
        const name = entry.info.name;
        
        if (!mergedData[name]) {
            mergedData[name] = {
                info: entry.info,
                Understanding: {},
                Reasoning: {},
                ReasoningAudio: {},
                ResponseText: {}
            };
        }
        
        // Merge Understanding data
        if (entry.Understanding) {
            mergedData[name].Understanding = entry.Understanding;
        }
        
        // Merge Reasoning data
        if (entry.Reasoning) {
            mergedData[name].Reasoning = entry.Reasoning;
        }
        
        // Merge Reasoning(Audio) data
        if (entry["Reasoning(Audio)"]) {
            mergedData[name].ReasoningAudio = entry["Reasoning(Audio)"];
        }
        
        // Merge Response(Text) data
        if (entry["Response(Text)"]) {
            mergedData[name].ResponseText = entry["Response(Text)"];
        }
    });
    
    return Object.values(mergedData);
}

// Calculate average score for ranking (using Understanding Acc as primary metric)
function calculateRankingScore(model) {
    const scores = [];
    
    if (model.Understanding.Acc && model.Understanding.Acc !== '-') {
        scores.push(parseFloat(model.Understanding.Acc));
    }
    if (model.Reasoning.Acc && model.Reasoning.Acc !== '-') {
        scores.push(parseFloat(model.Reasoning.Acc));
    }
    
    return scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
}

// Generate Understanding & Reasoning Table
function generateUnderstandingTable(data) {
    const mergedData = mergeLeaderboardData(data);
    
    // Sort by Understanding Acc + Reasoning Acc average
    mergedData.sort((a, b) => calculateRankingScore(b) - calculateRankingScore(a));
    
    const tbody = document.querySelector('#table-understanding tbody');
    tbody.innerHTML = '';
    
    mergedData.forEach((model, index) => {
        const row = document.createElement('tr');
        row.className = `model-${model.info.type}`;
        
        // Medal for top 3
        let medal = '';
        if (index === 0) medal = ' 🥇';
        else if (index === 1) medal = ' 🥈';
        else if (index === 2) medal = ' 🥉';
        
        // Model name with link
        const nameCell = model.info.link ? 
            `<a href="${model.info.link}" class="model-name-link" target="_blank">${model.info.name}${medal}</a>` :
            `${model.info.name}${medal}`;
        
        row.innerHTML = `
            <td class="rank-column">${index + 1}</td>
            <td>${nameCell}</td>
            <td>${model.Understanding.WER || '-'}</td>
            <td>${model.Understanding.SemSim || '-'}</td>
            <td class="metric-group-divider">${model.Understanding.Acc || '-'}</td>
            <td>${model.Reasoning.Acc || '-'}</td>
        `;
        
        tbody.appendChild(row);
    });
}

// Generate Reasoning (Audio) Table
function generateReasoningAudioTable(data) {
    const mergedData = mergeLeaderboardData(data);
    
    // Filter models that have ReasoningAudio data
    const filteredData = mergedData.filter(model => 
        Object.keys(model.ReasoningAudio).length > 0 &&
        model.ReasoningAudio.NISQA !== '-'
    );
    
    // Sort by NISQA score
    filteredData.sort((a, b) => {
        const scoreA = parseFloat(a.ReasoningAudio.NISQA) || 0;
        const scoreB = parseFloat(b.ReasoningAudio.NISQA) || 0;
        return scoreB - scoreA;
    });
    
    const tbody = document.querySelector('#table-reasoning-audio tbody');
    tbody.innerHTML = '';
    
    filteredData.forEach((model, index) => {
        const row = document.createElement('tr');
        row.className = `model-${model.info.type}`;
        
        let medal = '';
        if (index === 0) medal = ' 🥇';
        else if (index === 1) medal = ' 🥈';
        else if (index === 2) medal = ' 🥉';
        
        const nameCell = model.info.link ? 
            `<a href="${model.info.link}" class="model-name-link" target="_blank">${model.info.name}${medal}</a>` :
            `${model.info.name}${medal}`;
        
        row.innerHTML = `
            <td class="rank-column">${index + 1}</td>
            <td>${nameCell}</td>
            <td>${model.ReasoningAudio.NISQA || '-'}</td>
            <td>${model.ReasoningAudio.DNMOS || '-'}</td>
            <td>${model.ReasoningAudio.EmoAlign || '-'}</td>
            <td>${model.ReasoningAudio.VES || '-'}</td>
        `;
        
        tbody.appendChild(row);
    });
}

// Generate Response (Text) Table
function generateResponseTextTable(data) {
    const mergedData = mergeLeaderboardData(data);
    
    // Filter models that have ResponseText data
    const filteredData = mergedData.filter(model => 
        Object.keys(model.ResponseText).length > 0
    );
    
    // Sort by BERTScore
    filteredData.sort((a, b) => {
        const scoreA = parseFloat(a.ResponseText.BERTScore) || 0;
        const scoreB = parseFloat(b.ResponseText.BERTScore) || 0;
        return scoreB - scoreA;
    });
    
    const tbody = document.querySelector('#table-response-text tbody');
    tbody.innerHTML = '';
    
    filteredData.forEach((model, index) => {
        const row = document.createElement('tr');
        row.className = `model-${model.info.type}`;
        
        let medal = '';
        if (index === 0) medal = ' 🥇';
        else if (index === 1) medal = ' 🥈';
        else if (index === 2) medal = ' 🥉';
        
        const nameCell = model.info.link ? 
            `<a href="${model.info.link}" class="model-name-link" target="_blank">${model.info.name}${medal}</a>` :
            `${model.info.name}${medal}`;
        
        row.innerHTML = `
            <td class="rank-column">${index + 1}</td>
            <td>${nameCell}</td>
            <td>${model.ResponseText.BLEU || '-'}</td>
            <td>${model.ResponseText['ROUGE-L'] || '-'}</td>
            <td>${model.ResponseText.METEOR || '-'}</td>
            <td class="metric-group-divider">${model.ResponseText.BERTScore || '-'}</td>
            <td>${model.ResponseText.C1 || '-'}</td>
            <td>${model.ResponseText.C2 || '-'}</td>
            <td>${model.ResponseText.C3 || '-'}</td>
            <td>${model.ResponseText.C4 || '-'}</td>
        `;
        
        tbody.appendChild(row);
    });
}

// Tab switching functionality
function switchTab(tabName) {
    // Hide all content
    document.querySelectorAll('.leaderboard-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected content
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

// Load and initialize leaderboard
function initializeLeaderboard() {
    fetch('EchoMind_result.json')
        .then(response => response.json())
        .then(data => {
            generateUnderstandingTable(data);
            generateReasoningAudioTable(data);
            generateResponseTextTable(data);
        })
        .catch(error => console.error('Error loading leaderboard data:', error));
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initializeLeaderboard);