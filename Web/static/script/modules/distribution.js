import {useShared, Buffer, createWorker} from './helper.js'


const numWorkers = (useShared) ? 4 : 1;

let workersCreateKDE = [];
for (let i = 0; i < numWorkers; ++i)
    workersCreateKDE.push(createWorker(taskCreateKDE));


//const bandwidth = 1.0;


const gaussianDenom = Math.sqrt(2 * Math.PI);
const gaussianDenomInv = 1 / gaussianDenom;

function gaussian(x)
{
    return Math.exp(-(x*x/2)) * gaussianDenomInv;
}

function gaussianDeriv(x)
{
    return -Math.exp(-(x*x/2)) * x * gaussianDenomInv;
}

function epanechnikovKernel(x)
{
    if (x < -1 || x > 1)
        return 0;
    return (1 - x*x) * 3 / 4;
}

function uniformKernel(x)
{
    if (x < -1 || x > 1)
        return 0;
    return 0.5;
}

const kernel = gaussian;




function createKDE_Naive(values, count, bandwidth = 1.0)
{
    const maxValue = Math.max.apply(null, values) + 1;
    const step = maxValue / count;
    const n = values.length;
    const h = bandwidth;

    const kde = new Array(count);
    for (let x = 0, p = 0; p < count; x += step, ++p) {
        let sum = 0;
        for (let i = 0; i < n; ++i) {
            sum += kernel((x - values[i]) / h);
        }
        kde[p] = sum / (n*h);
    }

    return kde;
}


function createKDE_Accelerated(values, count, bandwidth = 1.0)
{
    const maxValue = Math.max.apply(null, values) + 1;
    const step = maxValue / count;
    const n = values.length;
    const h = bandwidth;
    const hInv = 1 / h;

    const kde = new Float32Array(new Buffer(count*4));
    for (let x = 0, p = 0; p < count; x += step, ++p) {
        let sum = 0;
        for (let i = 0; i < n; ++i) {
            const val = (x - values[i]) * hInv;
            sum += kernel(val);
        }
        kde[p] = sum / (n*h);
    }

    return kde;
}




function taskCreateKDE(e)
{
    const {values, kde, bandwidth, start, end} = e.data;

    const gaussianDenom = Math.sqrt(2 * Math.PI);
    const gaussianDenomInv = 1 / gaussianDenom;
    const gaussian = (x) => Math.exp(-(x*x/2)) * gaussianDenomInv;

    const maxValue = Math.max.apply(null, values) + 1;
    const step = maxValue / kde.length;
    const n = values.length;
    const h = bandwidth;
    const hInv = 1 / h;

    for (let x = step*start, p = start; p < end; x += step, ++p) {
        let sum = 0;
        for (let i = 0; i < n; ++i) {
            const val = (x - values[i]) * hInv;
            sum += gaussian(val);
        }
        kde[p] = sum / (n*h);
    }

    postMessage({kde: kde});
}


export async function createKDE(values, count, bandwidth = 1.0)
{
    let kde = new Float32Array(new Buffer(count*4));

    const chunkSize = Math.floor(count / numWorkers);
    let startChunk = [0];
    let endChunk = [chunkSize];
    for (let i = 1; i < numWorkers; ++i) {
        startChunk.push(endChunk[i-1]);
        endChunk.push(endChunk[i-1] + chunkSize);
    }
    endChunk[numWorkers-1] = count;

    const tasks = [];
    for (let i = 0; i < numWorkers; ++i)
    {
        tasks.push(new Promise(function (resolve) {
            workersCreateKDE[i].onmessage = (e) => {
                ({kde} = e.data);
                resolve();
            };
            workersCreateKDE[i].postMessage({
                values: values,
                kde: kde,
                bandwidth: bandwidth,
                start: startChunk[i],
                end: endChunk[i],
            });
        }));
    }

    await Promise.all(tasks);
    return kde;
}


function percentile(values, p)
{
    values = Array.from(values);
    values.sort((a, b) => a - b);
    const ij = (p / 100) * (values.length - 1);
    const i = Math.floor(ij);
    const jscale = ij - i;
    const iscale = 1 - jscale;
    if (i < 0 || i >= values.length)
        return 0;
    if (i+1 === values.length)
        return values[i];
    return (values[i] * iscale) + (values[i+1] * jscale);
}


export async function calculateDefaultSizeThresh(cellSizesSqrt)
{
    const kde = await createKDE(cellSizesSqrt, 500);
    const maxValue = Math.max.apply(null, cellSizesSqrt) + 1;
    const step = maxValue / kde.length;
    let idx = 1;
    for (; idx < kde.length-1; ++idx)
        if (kde[idx] < kde[idx-1] && kde[idx] < kde[idx+1])
            break;
    return (idx-1) * step;
}