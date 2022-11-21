function deepcopy(x) {
    return JSON.parse(JSON.stringify(x));
}

function mod(n, m) {
    return ((n % m) + m) % m;
}

async function initTSP() {
    tspCancelled = true;
    s = tf.randomUniform([M, M], 0, 1).div(1.2).round().arraySync();
    best_s = deepcopy(s);
    await drawTSP(s);
    smatrix(s);
}

function energy(s, gamma) {
    var d = 0;
    for (let x = 0; x < M; x++) {
        for (let i = 0; i < M; i++) {
            for (let y = 0; y < M; y++) {
                if (x != y) {
                    d += D[x][y] * s[x][i] * (s[y][mod((i+1), M)] + s[y][mod((i-1), M)]);
                }
            }
        }
    }
    var row_penalty = 0;
    for (let x = 0; x < M; x++) {
        row_penalty += (1 - tf.sum(s[x]).arraySync()) ** 2;
    }
    var col_penalty = 0;
    for (let i = 0; i < M; i++) {
        col_penalty += (1 - tf.sum(s.map(x => x[i])).arraySync()) ** 2;
    }
    return d + gamma * (row_penalty + col_penalty);
}

function connection_energy(x, i, s, gamma) {
    var d = 0;
    for (let y = 0; y < M; y++) {
        if (x != y) {
            d += D[x][y] * s[x][i] * (s[y][mod((i+1), M)] + s[y][mod((i-1), M)]);
        }
    }
    var row_penalty = (1 - tf.sum(s[x]).arraySync()) ** 2;
    var col_penalty = (1 - tf.sum(s.map(y => y[i])).arraySync()) ** 2;
    return d + gamma * (row_penalty + col_penalty);

}

async function optimize_step(beta, gamma) {
    for (const x of tf.util.createShuffledIndices(M)) {
        for (const i of tf.util.createShuffledIndices(M)) {
            v = s[x][i];
            complement = Number(!v);
            var s_prime = deepcopy(s);
            s_prime[x][i] = complement;
            var energy_delta = connection_energy(x, i, s_prime, gamma) - connection_energy(x, i, s, gamma);
            var switch_prob = 1 / (1 + Math.exp(beta * energy_delta));
            if (energy_delta < 0 || Math.random() < switch_prob) {
                s = s_prime;
                var energy_prime = energy(s_prime, 10);
                if (energy_prime < best_energy) {
                    best_energy = energy_prime;
                    best_s = deepcopy(s);
                }
                await drawTSP(s);
            }
        }
    }
    smatrix(s);
}

async function optimize() {
    tspCancelled = false;
    var gamma = M/2 - 0.5;
    var max_iter = 100;
    let beta_low = 0.7;
    let beta_high = 0.9;
    let beta_step = (beta_high - beta_low) / max_iter;
    for (let beta = beta_low; beta < beta_high; beta += beta_step) {
        if (!tspCancelled) {
            await optimize_step(beta, gamma);
            gamma += 0.5 / max_iter;
        }
    }
    // enforce valid solution
    s = best_s;
    await drawTSP(s);
    smatrix(s);
    //console.log("done");
}

function smatrix(s) {
    let latex = `$$\\begin{pmatrix}
    `;
    for (let x = 0; x < M; x++) {
        for (let i = 0; i < M; i++) {
            latex = latex + s[x][i];
            if (i < M-1) {
                latex = latex + " & ";
            }
        }
        latex = latex + `\\\\
        `;
    }
    latex = latex + `
    \\end{pmatrix}$$`;
    let smatrixdiv = document.getElementById("smatrix");
    smatrixdiv.innerHTML = latex;
    // check if katex is loaded
    if (typeof renderMathInElement == "function") {
        renderMathInElement(smatrixdiv, {});
    }
}

async function drawTSP(s) {
    // base layer
    tspCtx.drawImage(tspBackground, 0, 0, tspCanvas.width, tspCanvas.height);

    // cities
    for (let i = 0; i < M; i++) {
        tspCtx.beginPath();
        tspCtx.arc(...cities[i].mul(tspCanvas.width).arraySync(), 2, 0, 2 * Math.PI, true);
        tspCtx.stroke();
    }

    // paths
    for (let i = 0; i < M; i++) {
        for (let x = 0; x < M; x++) {
            for (let y = 0; y < M; y++) {
                if (s[x][i] * s[y][mod(i+1, M)]) {
                    tspCtx.beginPath();
                    tspCtx.moveTo(...cities[x].mul(tspCanvas.width).arraySync());
                    tspCtx.lineTo(...cities[y].mul(tspCanvas.width).arraySync());
                    tspCtx.stroke();
                }
            }
        }
    }
    await new Promise(r => setTimeout(r, 2));
}