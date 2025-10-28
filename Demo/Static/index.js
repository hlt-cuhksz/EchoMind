// Helper function to format values
function formatValue(value) {
    if (value === "-" || value === null || value === undefined) {
        return "-";
    }
    if (typeof value === "number") {
        return value.toFixed(2);
    }
    return value;
}

// Generate Leaderboard 1: Understanding & Reasoning
function generateLeaderboard1(leaderboardData) {
    const tbody = document.querySelector('#leaderboard1 tbody');
    tbody.innerHTML = ""; // Clear existing rows

    // Calculate average score for each entry
    const processedData = leaderboardData.leaderboardData.map(entry => {
        const understandingAcc = parseFloat(entry.Understanding.Acc) || 0;
        const reasoningAcc = parseFloat(entry.Reasoning.Acc) || 0;
        const avgScore = (understandingAcc + reasoningAcc) / 2;
        return { ...entry, avgScore };
    });

    // Sort by average score (descending)
    processedData.sort((a, b) => b.avgScore - a.avgScore);

    // Generate table rows
    processedData.forEach((entry, index) => {
        const row = document.createElement('tr');
        
        // Determine rank class for top 3
        let rankClass = '';
        if (index === 0) rankClass = 'rank-1';
        else if (index === 1) rankClass = 'rank-2';
        else if (index === 2) rankClass = 'rank-3';

        // Create row HTML
        const rankCell = `<td class="rank-col ${rankClass}">${index + 1}</td>`;
        const nameCell = `<td class="model-name">${entry.info.name}</td>`;
        
        // Understanding metrics
        const understandingCells = `
            <td>${formatValue(entry.Understanding.WER)}</td>
            <td>${formatValue(entry.Understanding.SemSim)}</td>
            <td>${formatValue(entry.Understanding.Acc)}</td>
        `;
        
        // Reasoning metrics
        const reasoningCell = `<td>${formatValue(entry.Reasoning.Acc)}</td>`;
        
        // Response Audio metrics
        const audioCells = `
            <td>${formatValue(entry["Reasoning(Audio)"].NISQA)}</td>
            <td>${formatValue(entry["Reasoning(Audio)"].DNMOS)}</td>
            <td>${formatValue(entry["Reasoning(Audio)"].EmoAlign)}</td>
            <td>${formatValue(entry["Reasoning(Audio)"].VES)}</td>
        `;
        
        // Average score
        const avgCell = `<td class="avg-score">${formatValue(entry.avgScore)}</td>`;
        
        row.innerHTML = rankCell + nameCell + understandingCells + reasoningCell + audioCells + avgCell;
        tbody.appendChild(row);
    });
}

// Generate Leaderboard 2: Response Text Quality
function generateLeaderboard2(leaderboardData) {
    const tbody = document.querySelector('#leaderboard2 tbody');
    tbody.innerHTML = ""; // Clear existing rows

    // Calculate average score for each entry
    const processedData = leaderboardData.leaderboardData.map(entry => {
        const textMetrics = entry["Response(Text)"];
        const values = Object.values(textMetrics).filter(v => typeof v === 'number');
        const avgScore = values.length > 0 
            ? values.reduce((sum, val) => sum + val, 0) / values.length 
            : 0;
        return { ...entry, avgScore };
    });

    // Sort by average score (descending)
    processedData.sort((a, b) => b.avgScore - a.avgScore);

    // Generate table rows
    processedData.forEach((entry, index) => {
        const row = document.createElement('tr');
        
        // Determine rank class for top 3
        let rankClass = '';
        if (index === 0) rankClass = 'rank-1';
        else if (index === 1) rankClass = 'rank-2';
        else if (index === 2) rankClass = 'rank-3';

        // Create row HTML
        const rankCell = `<td class="rank-col ${rankClass}">${index + 1}</td>`;
        const nameCell = `<td class="model-name">${entry.info.name}</td>`;
        
        // Response Text metrics
        const textMetrics = entry["Response(Text)"];
        const textCells = `
            <td>${formatValue(textMetrics.BLEU)}</td>
            <td>${formatValue(textMetrics["ROUGE-L"])}</td>
            <td>${formatValue(textMetrics.METEOR)}</td>
            <td>${formatValue(textMetrics.BERTScore)}</td>
            <td>${formatValue(textMetrics.C1)}</td>
            <td>${formatValue(textMetrics.C2)}</td>
            <td>${formatValue(textMetrics.C3)}</td>
            <td>${formatValue(textMetrics.C4)}</td>
        `;
        
        // Average score
        const avgCell = `<td class="avg-score">${formatValue(entry.avgScore)}</td>`;
        
        row.innerHTML = rankCell + nameCell + textCells + avgCell;
        tbody.appendChild(row);
    });
}

// Load JSON data and generate leaderboards
function loadLeaderboard1() {
    fetch('Demo/Data/leaderboard_data_1.json')
        .then(response => response.json())
        .then(data => generateLeaderboard1(data))
        .catch(error => {
            console.error('Error loading leaderboard 1 data:', error);
            document.querySelector('#leaderboard1 tbody').innerHTML = 
                '<tr><td colspan="11" class="error-message">Error loading data. Please check the console.</td></tr>';
        });
}

function loadLeaderboard2() {
    fetch('Demo/Data/leaderboard_data_2.json')
        .then(response => response.json())
        .then(data => generateLeaderboard2(data))
        .catch(error => {
            console.error('Error loading leaderboard 2 data:', error);
            document.querySelector('#leaderboard2 tbody').innerHTML = 
                '<tr><td colspan="11" class="error-message">Error loading data. Please check the console.</td></tr>';
        });
}

// Initialize leaderboards when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    loadLeaderboard1();
    loadLeaderboard2();
});