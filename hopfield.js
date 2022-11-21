function toHopfield(a) {
    return a.mul(2).add(-1);
}

function setCanvas(s) {
    tf.browser.toPixels(s.mul(-1).add(1).div(2).reshape([SIZE, SIZE, 1]), canvas);
}

function reset() {
    state = tf.ones([SIZE*SIZE]).mul(-1);
    cancelled = true;
    setCanvas(state);
}

async function recall() {
    cancelled = false;
    var img = tf.tensor(Array.from(ctx.getImageData(0, 0, SIZE, SIZE).data)).div(255);
    state = toHopfield(img.stridedSlice([0], [4*SIZE*SIZE], [4])).mul(-1);
    var old_state;
    do {
        old_state = state;
        state = tf.matMul(
                    patterns.transpose(),
                    tf.softmax(
                        tf.matMul(patterns, old_state.reshape([SIZE*SIZE, 1])).reshape([N*10])
                    ).reshape([N*10, 1])
                ).reshape([SIZE*SIZE]).mul(10000).round().div(10000);
        await animate(old_state, state);
    } while (tf.losses.absoluteDifference(old_state, state).dataSync()[0] > 0.000001);
}

async function animate(old_s, new_s) {
    for (let i = 1; i <= 10; i++) {
        if (!cancelled) {
            setCanvas(tf.add(old_s.mul(1-i/10), new_s.mul(i/10)));
            await new Promise(r => setTimeout(r, 100));
        }
    }
    return;
}

// new position from mouse event
function setPosition(e) {
    displaySize = Math.round(getComputedStyle(document.querySelector('canvas')).width.slice(0, -2));
    canvasOffset = $(canvas).offset();
    pos.x = Math.round((e.clientX - canvasOffset.left) / displaySize * SIZE);
    pos.y = Math.round((e.clientY - canvasOffset.top) / displaySize * SIZE);
}

function draw(e) {
    // mouse left button must be pressed
    if (e.buttons !== 1) return;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.beginPath(); // begin

    ctx.moveTo(pos.x, pos.y); // from
    setPosition(e);
    ctx.lineTo(pos.x, pos.y); // to

    ctx.stroke(); // draw it!
}