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
        
        // Average score
        const avgCell = `<td class="avg-score">${formatValue(entry.avgScore)}</td>`;
        
        row.innerHTML = rankCell + nameCell + understandingCells + reasoningCell + avgCell;
        tbody.appendChild(row);
    });
}

// Generate Leaderboard Audio: Response Audio Quality (Sorted by VES)
function generateLeaderboardAudio(leaderboardData) {
    const tbody = document.querySelector('#leaderboard-audio tbody');
    tbody.innerHTML = ""; // Clear existing rows

    // Filter out entries with no VES score and sort by VES (descending)
    const processedData = leaderboardData.leaderboardData
        .filter(entry => {
            const ves = entry["Reasoning(Audio)"].VES;
            return ves !== "-" && ves !== null && ves !== undefined;
        })
        .map(entry => ({
            ...entry,
            vesScore: parseFloat(entry["Reasoning(Audio)"].VES) || 0
        }))
        .sort((a, b) => b.vesScore - a.vesScore);

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
        
        // Response Audio metrics
        const audioCells = `
            <td>${formatValue(entry["Reasoning(Audio)"].NISQA)}</td>
            <td>${formatValue(entry["Reasoning(Audio)"].DNMOS)}</td>
            <td>${formatValue(entry["Reasoning(Audio)"].EmoAlign)}</td>
            <td class="avg-score">${formatValue(entry["Reasoning(Audio)"].VES)}</td>
        `;
        
        row.innerHTML = rankCell + nameCell + audioCells;
        tbody.appendChild(row);
    });
}

// Generate Leaderboard 2: Response Text Quality (Sorted by C1 and C4 average)
function generateLeaderboard2(leaderboardData) {
    const tbody = document.querySelector('#leaderboard2 tbody');
    tbody.innerHTML = ""; // Clear existing rows

    // Calculate average of C1 and C4 for each entry
    const processedData = leaderboardData.leaderboardData.map(entry => {
        const textMetrics = entry["Response(Text)"];
        const c1 = parseFloat(textMetrics.C1) || 0;
        const c4 = parseFloat(textMetrics.C4) || 0;
        const avgScore = (c1 + c4) / 2;
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
        
        // Average score (C1 + C4) / 2
        const avgCell = `<td class="avg-score">${formatValue(entry.avgScore)}</td>`;
        
        row.innerHTML = rankCell + nameCell + textCells + avgCell;
        tbody.appendChild(row);
    });
}

// Load JSON data and generate leaderboards
function loadLeaderboard1() {
    fetch('./Demo/Data/leaderboard_data_1.json')
        .then(response => response.json())
        .then(data => {
            generateLeaderboard1(data);
            generateLeaderboardAudio(data);  // 使用相同数据生成 Audio 排行榜
        })
        .catch(error => {
            console.error('Error loading leaderboard 1 data:', error);
            document.querySelector('#leaderboard1 tbody').innerHTML = 
                '<tr><td colspan="7" class="error-message">Error loading data. Please check the console.</td></tr>';
            document.querySelector('#leaderboard-audio tbody').innerHTML = 
                '<tr><td colspan="6" class="error-message">Error loading data. Please check the console.</td></tr>';
        });
}

function loadLeaderboard2() {
    fetch('./Demo/Data/leaderboard_data_2.json')
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
    loadLeaderboard1();  // 这会同时生成 leaderboard1 和 leaderboard-audio
    loadLeaderboard2();
});