import express from 'express';
import fetch from 'node-fetch';
import os from 'os';

// --- CONFIGURATION ---
const HOST = '0.0.0.0';
const PORT = process.env.PORT || 8000;
const POLL_INTERVAL = 5000;
const RETRY_DELAY = 5000;
const MAX_HISTORY = 500;
const MIN_HISTORY_FOR_PREDICTION = 15; // Cần nhiều dữ liệu hơn cho các thuật toán phức tạp

// --- GLOBAL STATE ---
let latest_result_100 = { Phien: 0, Xuc_xac_1: 0, Xuc_xac_2: 0, Xuc_xac_3: 0, Tong: 0, Ket_qua: "Chưa có" };
let latest_result_101 = { Phien: 0, Xuc_xac_1: 0, Xuc_xac_2: 0, Xuc_xac_3: 0, Tong: 0, Ket_qua: "Chưa có" };

let history_100 = [];
let history_101 = [];
let last_sid_100 = null;
let last_sid_101 = null;
let sid_for_tx = null;

const logger = {
    info: (msg) => console.log(`[${new Date().toISOString()}] [INFO] ${msg}`),
    error: (msg) => console.error(`[${new Date().toISOString()}] [ERROR] ${msg}`),
};

// --- ADVANCED PREDICTION ENGINE ---

// Performance tracker for each algorithm
const algorithmPerformance = {
    markov: { wins: 0, predictions: 0, weight: 1.0, last_prediction: null },
    bayesian: { wins: 0, predictions: 0, weight: 1.0, last_prediction: null },
    momentum: { wins: 0, predictions: 0, weight: 1.0, last_prediction: null },
    logistic: { wins: 0, predictions: 0, weight: 1.0, last_prediction: null },
    entropy: { wins: 0, predictions: 0, weight: 1.0, last_prediction: null },
    pattern: { wins: 0, predictions: 0, weight: 1.0, last_prediction: null },
    trend: { wins: 0, predictions: 0, weight: 1.0, last_prediction: null },
};

// --- HELPER FUNCTIONS ---
const getResultsList = (history) => history.map(h => h.Ket_qua).filter(kq => ['Tài', 'Xỉu'].includes(kq));
const getStreaks = (results) => {
    if (!results || results.length === 0) return [];
    const streaks = [];
    let current_streak_type = results[0];
    let current_streak_length = 0;
    for (const result of results) {
        if (result === current_streak_type) {
            current_streak_length++;
        } else {
            streaks.push({ type: current_streak_type, length: current_streak_length });
            current_streak_type = result;
            current_streak_length = 1;
        }
    }
    streaks.push({ type: current_streak_type, length: current_streak_length });
    return streaks;
};

// --- ALGORITHMS ---

/** 1. MARKOV CHAIN: Predicts based on the probability of transitioning from the last state. */
function markovChain(results) {
    if (results.length < 10) return null;
    const transitions = { 'Tài': { 'Tài': 0, 'Xỉu': 0 }, 'Xỉu': { 'Tài': 0, 'Xỉu': 0 } };
    for (let i = 1; i < results.length; i++) {
        const from = results[i];
        const to = results[i - 1];
        if (transitions[from] && transitions[from][to] !== undefined) {
            transitions[from][to]++;
        }
    }
    const lastResult = results[0];
    const nextStates = transitions[lastResult];
    if (nextStates['Tài'] > nextStates['Xỉu']) return 'Tài';
    if (nextStates['Xỉu'] > nextStates['Tài']) return 'Xỉu';
    return null;
}

/** 2. BAYESIAN INFERENCE (Simplified): Predicts based on patterns of the last 2 results. */
function bayesianPrediction(results) {
    if (results.length < 10) return null;
    const lastTwo = results.slice(0, 2).join('');
    if (lastTwo.length < 2) return null;

    let tàiCount = 0, xỉuCount = 0;
    for (let i = 2; i < results.length; i++) {
        if (results[i] === lastTwo[1] && results[i - 1] === lastTwo[0]) {
            if (results[i - 2] === 'Tài') tàiCount++;
            else xỉuCount++;
        }
    }
    if (tàiCount > xỉuCount) return 'Tài';
    if (xỉuCount > tàiCount) return 'Xỉu';
    return null;
}

/** 3. MOMENTUM: Compares short-term trend vs long-term trend. */
function momentum(results) {
    if (results.length < 15) return null;
    const shortTerm = results.slice(0, 5);
    const longTerm = results.slice(0, 15);
    const shortMomentum = shortTerm.filter(r => r === 'Tài').length - shortTerm.filter(r => r === 'Xỉu').length;
    const longMomentum = longTerm.filter(r => r === 'Tài').length - longTerm.filter(r => r === 'Xỉu').length;
    
    // If short-term trend is significantly stronger than long-term, follow it
    if (shortMomentum > longMomentum + 2) return 'Tài';
    if (shortMomentum < longMomentum - 2) return 'Xỉu';
    
    // If momentum is strong in one direction
    if (shortMomentum >= 3) return 'Tài';
    if (shortMomentum <= -3) return 'Xỉu';
    return null;
}

/** 4. LOGISTIC REGRESSION (Proxy): A weighted model of different factors. */
function logisticRegression(results) {
    if (results.length < 10) return null;
    const streaks = getStreaks(results);
    const last10 = results.slice(0, 10);
    
    const f1_currentStreak = streaks[0].length;
    const f2_isBệt = f1_currentStreak >= 3 ? (streaks[0].type === 'Tài' ? 1 : -1) : 0;
    const f3_balance = last10.filter(r => r === 'Tài').length - 5; // Positive for more Tài
    let f4_switches = 0;
    for (let i = 1; i < last10.length; i++) if (last10[i] !== last10[i-1]) f4_switches++;
    
    // Weights (pre-determined logic)
    const w1 = 0.2; // Streak length
    const w2 = 0.4; // Strong bệt signal
    const w3 = 0.3; // Recent balance
    const w4 = -0.1; // High switches suggest instability, slightly negative weight
    
    const score = (f1_currentStreak * w1 * (streaks[0].type === 'Tài' ? 1 : -1)) + (f2_isBệt * w2) + (f3_balance * w3) + (f4_switches * w4);
    
    return score > 0.2 ? 'Tài' : (score < -0.2 ? 'Xỉu' : null);
}

/** 5. ENTROPY ANALYSIS: Predicts based on the randomness of the sequence. */
function entropyAnalysis(results) {
    if (results.length < 20) return null;
    const window = results.slice(0, 20);
    const p_tài = window.filter(r => r === 'Tài').length / window.length;
    if (p_tài === 0 || p_tài === 1) return results[0]; // Pure streak, follow it
    
    const p_xỉu = 1 - p_tài;
    const entropy = -p_tài * Math.log2(p_tài) - p_xỉu * Math.log2(p_xỉu);
    
    if (entropy < 0.75) { // Low entropy = predictable streak
        return p_tài > p_xỉu ? 'Tài' : 'Xỉu';
    } else if (entropy > 0.98) { // High entropy = random, likely 1-1
        return results[0] === 'Tài' ? 'Xỉu' : 'Tài';
    }
    return null;
}

/** 6. ADVANCED PATTERN MATCHER: Finds fixed and dynamic repeating patterns. */
function advancedPatternMatcher(results) {
    const streaks = getStreaks(results);
    if (streaks.length < 4) return null;

    const streak_lengths = streaks.map(s => s.length);
    const last_result = results[0];
    
    // Fixed patterns dictionary
    const fixed_patterns = {
        '1-1': [1, 1], '2-2': [2, 2], '1-2-1': [1, 2, 1], '2-1-2': [2, 1, 2],
        '3-2-1': [3, 2, 1], '1-2-3': [1, 2, 3], '3-1-3': [3, 1, 3], '4-1-4': [4, 1, 4],
        '2-1-1-2': [2, 1, 1, 2], '1-1-2-2': [1, 1, 2, 2]
    };
    for (const [name, pattern_def] of Object.entries(fixed_patterns)) {
        if (streak_lengths.length >= pattern_def.length) {
            if (JSON.stringify(streak_lengths.slice(0, pattern_def.length)) === JSON.stringify(pattern_def.reverse())) {
                return last_result === 'Tài' ? 'Xỉu' : 'Tài'; // Break pattern
            }
        }
    }
    
    // Dynamic pattern detection (look for ABAB or ABCABC)
    if (streak_lengths.length >= 4 && streak_lengths[0] === streak_lengths[2] && streak_lengths[1] === streak_lengths[3]) {
        return last_result === 'Tài' ? 'Xỉu' : 'Tài'; // Break ABAB
    }
    if (streak_lengths.length >= 6 && streak_lengths[0] === streak_lengths[3] && streak_lengths[1] === streak_lengths[4] && streak_lengths[2] === streak_lengths[5]) {
         return last_result === 'Tài' ? 'Xỉu' : 'Tài'; // Break ABCABC
    }

    return null;
}

/** 7. TREND ANALYSIS: Identifies the dominant trend. */
function trendAnalysis(results) {
    const streaks = getStreaks(results);
    if (streaks.length < 2) return null;

    const currentStreak = streaks[0];
    const last15 = results.slice(0, 15);
    const taiCount = last15.filter(r => r === 'Tài').length;
    const xiuCount = 15 - taiCount;
    
    if (currentStreak.length >= 5) { // Xu hướng bệt dài
        return currentStreak.type;
    }
    if (taiCount - xiuCount >= 7) { // Xu hướng nghiêng Tài
        return 'Tài';
    }
    if (xiuCount - taiCount >= 7) { // Xu hướng nghiêng Xỉu
        return 'Xỉu';
    }
    
    let switches = 0;
    for (let i=0; i < streaks.length - 1; i++) {
        if(streaks[i].length === 1) switches++;
    }
    if (switches >= 3 && currentStreak.length === 1) { // Xu hướng cầu nhảy (1-1)
        return results[0] === 'Tài' ? 'Xỉu' : 'Tài';
    }

    return null;
}

/** Updates performance metrics based on the last actual result. */
function updatePerformanceMetrics(actualResult) {
    for (const key in algorithmPerformance) {
        const alg = algorithmPerformance[key];
        if (alg.last_prediction !== null) {
            if (alg.last_prediction === actualResult) {
                alg.wins++;
            }
            alg.predictions++;
            // Update weight using Laplace smoothing to avoid zero probability
            alg.weight = (alg.wins + 1) / (alg.predictions + 2);
        }
        alg.last_prediction = null; // Reset for next round
    }
}

/** Main function to get a combined, weighted prediction. */
function getCombinedPrediction(history) {
    const results = getResultsList(history);

    if (results.length < MIN_HISTORY_FOR_PREDICTION) {
        return { prediction: `Đang chờ đủ dữ liệu... (${results.length}/${MIN_HISTORY_FOR_PREDICTION})`, confidence: 0, trend: "Chưa xác định", voters: [] };
    }

    // Call all algorithms
    const predictions = {
        markov: markovChain(results),
        bayesian: bayesianPrediction(results),
        momentum: momentum(results),
        logistic: logisticRegression(results),
        entropy: entropyAnalysis(results),
        pattern: advancedPatternMatcher(results),
        trend: trendAnalysis(results),
    };

    const votes = { 'Tài': 0, 'Xỉu': 0 };
    const voters = [];
    let totalWeight = 0;

    // Tally weighted votes
    for (const [algName, prediction] of Object.entries(predictions)) {
        if (prediction) {
            const weight = algorithmPerformance[algName].weight;
            votes[prediction] += weight;
            totalWeight += weight;
            voters.push({ alg: algName, vote: prediction, weight: weight.toFixed(2) });
            // Store this prediction for the next performance update
            algorithmPerformance[algName].last_prediction = prediction;
        }
    }
    
    let final_prediction;
    let confidence;
    let winning_weight = 0;

    if (votes['Tài'] > votes['Xỉu']) {
        final_prediction = 'Tài';
        winning_weight = votes['Tài'];
    } else if (votes['Xỉu'] > votes['Tài']) {
        final_prediction = 'Xỉu';
        winning_weight = votes['Xỉu'];
    } else {
        final_prediction = results[0] === "Tài" ? "Xỉu" : "Tài"; // Fallback: break 1-1
        algorithmPerformance['markov'].last_prediction = final_prediction; // Assign to one alg for tracking
    }
    
    if (totalWeight > 0) {
        confidence = (winning_weight / totalWeight) * 100;
    } else {
        confidence = 40; // Base confidence for fallback
    }

    // Determine current trend for display
    const currentTrend = trendAnalysis(results) || (getStreaks(results)[0].length >=3 ? `Bệt ${getStreaks(results)[0].type}`: "Cầu hỗn hợp");

    return {
        prediction: final_prediction,
        confidence: Math.round(Math.min(confidence, 98)),
        trend: currentTrend,
        voters: voters,
    };
}

// --- POLLING LOGIC ---
const pollApi = async (gid, result_store, history, is_md5) => {
    const url = `https://jakpotgwab.geightdors.net/glms/v1/notify/taixiu?platform_id=g8&gid=${gid}`;
    while (true) {
        try {
            const resp = await fetch(url, { headers: { 'User-Agent': 'NodeJS-Proxy/1.0' }, timeout: 10000 });
            const data = await resp.json();

            if (data?.status === 'OK' && Array.isArray(data.data)) {
                for (const game of data.data) {
                    if (!is_md5 && game.cmd === 1008) sid_for_tx = game.sid;
                }

                for (const game of data.data) {
                    let is_new_result = false;
                    let sid, d1, d2, d3;

                    if (is_md5 && game.cmd === 2006) {
                        ({ sid, d1, d2, d3 } = game);
                        if (sid && sid !== last_sid_101 && d1 != null && d2 != null && d3 != null) {
                            last_sid_101 = sid; is_new_result = true;
                        }
                    } else if (!is_md5 && game.cmd === 1003) {
                        sid = sid_for_tx;
                        ({ d1, d2, d3 } = game);
                        if (sid && sid !== last_sid_100 && d1 != null && d2 != null && d3 != null) {
                            last_sid_100 = sid; is_new_result = true; sid_for_tx = null;
                        }
                    }

                    if (is_new_result) {
                        const total = d1 + d2 + d3;
                        const ket_qua = total <= 10 ? "Xỉu" : "Tài";
                        const result = {
                            Phien: sid, Xuc_xac_1: d1, Xuc_xac_2: d2, Xuc_xac_3: d3,
                            Tong: total, Ket_qua: ket_qua, timestamp: new Date().toISOString()
                        };
                        
                        // IMPORTANT: Update performance BEFORE adding new result to history
                        if (is_md5) {
                            updatePerformanceMetrics(ket_qua);
                        }

                        updateResult(result_store, history, result);
                        logger.info(`[${is_md5 ? "MD5" : "TX"}] Phiên ${sid} - Kết quả: ${ket_qua}`);
                    }
                }
            }
        } catch (e) {
            logger.error(`Lỗi API ${gid}: ${e.message}`);
            await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
        }
        await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
    }
};

const updateResult = (store, history, result) => {
    Object.assign(store, result);
    history.unshift({ ...result });
    if (history.length > MAX_HISTORY) history.pop();
};

// --- EXPRESS API ENDPOINTS ---
const app = express();
app.use(express.json());

app.get("/", (req, res) => res.send("API Server for TaiXiu is running (Advanced Node.js version)."));

app.get("/api/taixiu", (req, res) => res.json(latest_result_100));

app.get("/api/taixiumd5", (req, res) => {
    const current_session_data = { ...latest_result_101 };
    const prediction_result = getCombinedPrediction(history_101);
    const current_session_id = current_session_data.Phien || 0;
    
    const history_string = history_101.map(h => h.Ket_qua?.[0]).slice(0, 20).join('');

    const response_data = {
        "Phien_Hien_Tai": {
            "Phien": current_session_id,
            "Ket_Qua": current_session_data.Ket_qua
        },
        "Du_Doan_Phien_Tiep_Theo": {
            "Phien_Du_Doan": current_session_id > 0 ? current_session_id + 1 : "Chờ phiên",
            "Du_Doan": prediction_result.prediction,
            "Do_Tin_Cay": `${prediction_result.confidence}%`,
            "Xu_Huong_Hien_Tai": prediction_result.trend,
            "Chuoi_Ket_Qua": history_string,
            "Thuat_Toan_Bo_Phieu": prediction_result.voters,
        }
    };
    res.json(response_data);
});

// RE-ADDED PERFORMANCE ENDPOINT
app.get("/api/performance/md5", (req, res) => {
    const performance_data = Object.entries(algorithmPerformance).map(([name, stats]) => ({
        Thuat_Toan: name,
        So_Lan_Thang: stats.wins,
        Tong_So_Lan_Doan: stats.predictions,
        Ty_Le_Thang: stats.predictions > 0 ? `${((stats.wins / stats.predictions) * 100).toFixed(2)}%` : 'N/A',
        He_So_Dieu_Chinh: stats.weight.toFixed(4) // Trọng số hiện tại
    }));
    res.json({
        title: "Hiệu Suất và Hệ Số Điều Chỉnh của các Thuật Toán Dự Đoán",
        data: performance_data
    });
});

app.get("/api/lichsumd5", (req, res) => {
    res.json(history_101.slice(0, 100));
});


// --- MAIN EXECUTION ---
const startServer = () => {
    logger.info("Khởi động hệ thống API Tài Xỉu (Phiên bản Nâng cao)...");
    
    pollApi("vgmn_100", latest_result_100, history_100, false);
    pollApi("vgmn_101", latest_result_101, history_101, true);
    
    logger.info("Đã bắt đầu polling dữ liệu.");
    
    app.listen(PORT, HOST, () => {
        logger.info(`Server đang chạy tại http://${HOST}:${PORT}`);
    });
};

startServer();