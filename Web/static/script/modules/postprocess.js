import {useShared, Buffer, createWorker} from './helper.js'


const numWorkers = (useShared) ? 4 : 1;
const workersCreatePosNegMask = [];
const workersMarkBackground = [];
const workersComputeCellMapping = [];
for (let i = 0; i < numWorkers; ++i) {
    workersCreatePosNegMask.push(createWorker(taskCreatePosNegMask));
    workersMarkBackground.push(createWorker(taskMarkBackground));
    workersComputeCellMapping.push(createWorker(taskComputeCellMapping));
}


// imageData object (from main controls file)
// cellData object (has map, info, outlines)



export function toGrayscale(image)
{
    let gray = new Uint8Array(new Buffer(image.width*image.height));
    for (let i = 0, j = 0; i < gray.length; ++i, j += 4)
        gray[i] = Math.round(image.data[j]*0.299 + image.data[j+1]*0.587 + image.data[j+2]*0.114);
    return gray;
}


export function toGrayscaleMax(image)
{
    let gray = new Uint8Array(new Buffer(image.width*image.height));
    for (let i = 0, j = 0; i < gray.length; ++i, j += 4)
        gray[i] = Math.max(image.data[j], image.data[j+1], image.data[j+2]);
    return gray;
}




export function calcOpticalDensity(image)
{
    const lut = new Float32Array(256);
    for (let i = 1; i < 256; i+= 1)
        lut[i] = Math.log10(255 / i);
    lut[0] = lut[1];

    const opticalDensity = new Uint8Array(new Buffer(image.width*image.height));
    for (let i = 0, j = 0; i < opticalDensity.length; ++i, j += 4)
        opticalDensity[i] = Math.round((lut[image.data[j]] + lut[image.data[j+1]] + lut[image.data[j+2]]) * 35.3);
    return opticalDensity;
}




// Takes a single channel array of grayscale pixels.
// Creates a histogram of 0-255 values.
// Also, have max value in [256] and max count in [257].
export function createHistogram(channel)
{
    let counts = new Array(258).fill(0);
    for (let i = 0; i < channel.length; ++i)
        counts[channel[i]]++;

    // find max value
    for (let i = 255; i >= 0; --i)
        if (counts[i] > 0) {
            counts[256] = i;
            break;
        }

    // find max count
    for (let i = 0; i < 256; ++i)
        if (counts[i] > counts[257])
            counts[257] = counts[i];

    return counts;
}




export function calculateDisplayRange(hist)
{
    let countSum = 0;
    for (let i = 0; i < 256; ++i)
        countSum += hist[i];

    let minIndex = 0;
    let maxIndex = 255;
    if (hist[0] > hist[1]) {
        minIndex++;
        countSum -= hist[0];
    }
    if (hist[255] > hist[254]) {
        maxIndex--;
        countSum -= hist[255];
    }

    let countMax = countSum *= 0.001;
    let minDisplay = 0;
    for (let i = 0; i < 256; ++i)
        if (hist[i] != 0) {
            minDisplay = i;
            break;
        }
    let maxDisplay = hist[256];

    let count = countMax;
    for (let i = minIndex; i < 256; ++i) {
        let nextCount = hist[i];
        if (count < nextCount) {
            minDisplay = i;
            break;
        }
        count -= nextCount;
    }

    count = countMax;
    for (let i = maxIndex; i >= 0; --i) {
        let nextCount = hist[i];
        if (count < nextCount) {
            maxDisplay = i;
            break;
        }
        count -= nextCount;
    }

    return [minDisplay, maxDisplay];
}




function calculateChunks(thewidth, theheight)
{
    const chunkSize = Math.floor(theheight / numWorkers) * thewidth;
    const start = [0], end = [chunkSize];
    for (let i = 1; i < numWorkers; ++i) {
        start.push(end[i-1]);
        end.push(end[i-1] + chunkSize);
    }
    end[numWorkers-1] = thewidth * theheight;
    return {start: start, end: end};
}




export function extractSegChannels(origImage, segImage)
{
    const [red, green, blue] = extractRGBChannels(segImage);
    const gray = toGrayscale(origImage);
    //applyOriginalImageIntensity(gray, red);
    //applyOriginalImageIntensity(gray, blue);
    return {segRed: red, segGreen: green, segBlue: blue};
}


function extractRGBChannels(image)
{
    const numPixels = image.width * image.height;
    const red = new Uint8Array(new Buffer(numPixels));
    const green = new Uint8Array(new Buffer(numPixels));
    const blue = new Uint8Array(new Buffer(numPixels));

    for (let i = 0, j = 0; i < numPixels; ++i, j += 4) {
        red[i] = image.data[j];
        green[i] = image.data[j+1];
        blue[i] = image.data[j+2];
    }

    return [red, green, blue];
}


/*
function applyOriginalImageIntensity(gray, channel, orig_image_intensity_effect = 0.15)
{
    for (let i = 0; i < gray.length; ++i) {
        let val = (channel[i] > 10) ? gray[i] * orig_image_intensity_effect : 0;
        val += channel[i];
        channel[i] = (val > 255) ? 255 : val;
    }
}
*/




function taskCreatePosNegMask(e)
{
    const {red, green, blue, mask, posneg, thresh, start, end} = e.data;

    for (let i = start; i < end; ++i) {
        if (mask != null && mask[i] != 255)
            continue;
        const rb = red[i] + blue[i];
        if (rb > thresh && green[i] <= 80 && red[i] >= blue[i])
            posneg[i] = 1;
        else if (rb > thresh && green[i] <= 80 && blue[i] > red[i])
            posneg[i] = -1;
    }

    postMessage({posneg: posneg});
}


async function createPosNegMask(data, mask, segThresh)
{
    let posneg = new Int8Array(new Buffer(data.size));
    const {start, end} = calculateChunks(data.width, data.height);

    const tasks = [];
    for (let i = 0; i < numWorkers; ++i)
    {
        tasks.push(new Promise(function (resolve) {
            workersCreatePosNegMask[i].onmessage = (e) => {
                ({posneg} = e.data);
                resolve();
            };
            workersCreatePosNegMask[i].postMessage({
                red: data.segRed,
                green: data.segGreen,
                blue: data.segBlue,
                mask: mask,
                posneg: posneg,
                thresh: segThresh,
                start: start[i],
                end: end[i],
            });
        }));
    }

    await Promise.all(tasks);
    return posneg;
}




function taskMarkBackground(e)
{
    const {width, height, posneg, seeds} = e.data;

    const neigh_x = [0, -1, 1, 0];
    const neigh_y = [-1, 0, 0, 1];
    let pixels_x = [];
    let pixels_y = [];

    for (let i = 0; i < seeds.length; i += 2) {
        const idx = seeds[i+1]*width + seeds[i];
        if (posneg[idx] == 0) {
            pixels_x.push(seeds[i]);
            pixels_y.push(seeds[i+1]);
            posneg[idx] = 100;
        }
    }

    while (pixels_x.length > 0)
    {
        const x = pixels_x.pop();
        const y = pixels_y.pop();
        for (let n = 0; n < neigh_x.length; ++n)
        {
            const nx = x + neigh_x[n];
            const ny = y + neigh_y[n];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const ni = ny*width + nx;
                if (posneg[ni] == 0) {
                    pixels_x.push(nx);
                    pixels_y.push(ny);
                    posneg[ni] = 100;
                }
            }
        }
    }

    postMessage({posneg: posneg});
}


async function markBackground(data, posneg)
{
    const border = [];
    for (let x = 0; x < data.width; ++x) {
        border.push(x);
        border.push(0);
        border.push(x);
        border.push(data.height-1);
    }
    for (let y = 1; y < data.height-1; ++y) {
        border.push(0);
        border.push(y);
        border.push(data.width-1);
        border.push(y);
    }
    const lengthPerWorker = Math.floor((border.length/2) / numWorkers) * 2;
    const seeds = [];
    for (let i = 0; i < numWorkers-1; ++i)
        seeds.push(border.slice(i*lengthPerWorker, i*lengthPerWorker+lengthPerWorker));
    seeds.push(border.slice((numWorkers-1)*lengthPerWorker));

    const tasks = [];
    for (let i = 0; i < numWorkers; ++i)
    {
        tasks.push(new Promise(function (resolve) {
            workersMarkBackground[i].onmessage = (e) => {
                ({posneg} = e.data);
                resolve();
            };
            workersMarkBackground[i].postMessage({
                width: data.width,
                height: data.height,
                posneg: posneg,
                seeds: seeds[i],
            });
        }));
    }

    await Promise.all(tasks);
    return posneg;
}




function taskComputeCellMapping(e)
{
    const {posneg, marker, width, height, start, end} = e.data;

    const neigh_x = [-1, 0, 1, -1, 1, -1, 0, 1];
    const neigh_y = [-1, -1, -1, 0, 0, 1, 1, 1];
    const neigh_i = [-width-1, -width, -width+1, -1, 1, width-1, width, width+1];
    let cluster_prob;
    const pixels_i = [];
    const pixels_x = [];
    const pixels_y = [];
    const cellBorders = [];
    const cellSizesSqrt = [];

    for (let idx = start; idx < end; ++idx)
    {
        if (posneg[idx] == 1 || posneg[idx] == -1)
        {
            let x = idx % width;
            let y = Math.floor(idx / width);
            let cluster = [idx];
            cluster_prob = posneg[idx];
            pixels_i.push(idx);
            posneg[idx] = 50;
            let borderSet = new Set();
            borderSet.add(idx);
            for (let n = 0; n < neigh_i.length; ++n) {
                let nx = x + neigh_x[n];
                let ny = y + neigh_y[n];
                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                    borderSet.add(idx + neigh_i[n]);
            }
            //let maxMarker = 0;
            let markerVal = 0;
            count = 0;

            while (pixels_i.length > 0)
            {
                let i = pixels_i.pop();
                x = i % width;
                y = Math.floor(i / width);
                //if (marker[i] > maxMarker)
                //    maxMarker = marker[i];
                markerVal += marker[i]
                count += 1;

                for (let n = 0; n < neigh_i.length; ++n)
                {
                    let nx = x + neigh_x[n];
                    let ny = y + neigh_y[n];
                    let ni = i + neigh_i[n];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                        continue;
                    if (posneg[ni] == 1 || posneg[ni] == -1 || posneg[ni] == 0) {
                        cluster.push(ni);
                        pixels_i.push(ni);
                        cluster_prob += posneg[ni];
                        posneg[ni] = 50;
                        if ((ny > 0 && posneg[ni-width] == 100) || (nx > 0 && posneg[ni-1] == 100) || (nx < width-1 && posneg[ni+1]) == 100 || (ny < height-1 && posneg[ni+width] == 100)) {
                            borderSet.add(ni);
                            for (let nn = 0; nn < neigh_i.length; ++nn) {
                                let nnx = nx + neigh_x[nn];
                                let nny = ny + neigh_y[nn];
                                if (nnx >= 0 && nnx < width && nny >= 0 && nny < height)
                                    borderSet.add(ni + neigh_i[nn]);
                            }
                        }
                    }
                }
            }

            let cluster_value = (cluster_prob < 0) ? -2 : 2;
            for (const index of cluster)
                posneg[index] = cluster_value;

            markerVal = Math.round(markerVal / count);

            //let border = Array.from(borderSet);
            //border.sort((a, b) => a - b);
            //cellBorders.push(cluster.length, (cluster_prob < 0) ? 0 : 1, maxMarker, [...border]);

            //cellBorders.push(cluster.length, (cluster_prob < 0) ? 0 : 1, maxMarker, Array.from(borderSet));
            cellBorders.push(cluster.length, (cluster_prob < 0) ? 0 : 1, markerVal, Array.from(borderSet));
            cellSizesSqrt.push(Math.sqrt(cluster.length));
        }
    }

    // put background back to 0
    //for (let i = 0; i < posneg.length; ++i)
    //    if (posneg[i] == 100)
    //        posneg[i] = 0;

    postMessage({cellBorders: cellBorders, posneg: posneg, cellSizesSqrt: cellSizesSqrt});
}


async function computeCellMapping(data, posneg)
{
    const cellBorders = [];
    const cellSizesSqrt = [];
    const {start, end} = calculateChunks(data.width, data.height);

    let tasks = [];
    for (let i = 1; i < numWorkers; ++i)
    {
        tasks.push(new Promise(function (resolve) {
            workersComputeCellMapping[i].onmessage = (e) => {
                //console.log('lengths:', e.data.cellBorders.length, e.data.cellSizesSqrt.length);
                cellBorders.push(...e.data.cellBorders);
                cellSizesSqrt.push(...e.data.cellSizesSqrt);
                resolve();
            };
            workersComputeCellMapping[i].postMessage({
                shared: useShared,
                posneg: posneg,
                marker: data.marker,
                width: data.width,
                height: data.height,
                start: start[i]-data.width,
                end: start[i],
            });
        }));
    }

    await Promise.all(tasks);

    tasks = [];
    for (let i = 0; i < numWorkers; ++i)
    {
        tasks.push(new Promise(function (resolve) {
            workersComputeCellMapping[i].onmessage = (e) => {
                //console.log('lengths:', e.data.cellBorders.length, e.data.cellSizesSqrt.length);
                //cellBorders.push(...e.data.cellBorders);
                for (const cb of e.data.cellBorders)
                    cellBorders.push(cb);
                //cellSizesSqrt.push(...e.data.cellSizesSqrt);
                for (const cs of e.data.cellSizesSqrt)
                    cellSizesSqrt.push(cs);
                ({posneg} = e.data);
                resolve();
            };
            workersComputeCellMapping[i].postMessage({
                posneg: posneg,
                marker: data.marker,
                width: data.width,
                height: data.height,
                start: start[i],
                end: end[i],
            });
        }));
    }

    await Promise.all(tasks);

    //taskComputeCellMapping returns cellData
    //cellInfo and cellOutlines

    let count = 0; // number of total outline pixels
    for (let i = 3; i < cellBorders.length; i += 4)
        count += cellBorders[i].length;
    let cellData = new Uint32Array(new Buffer(cellBorders.length * 5));
    let cellOutlines = new Uint32Array(new Buffer(count*4));
    let k = 0;
    for (let i = 0, j = 0; i < cellBorders.length; i += 4, j += 5) {
        cellData[j] = cellBorders[i];        // cell size (area, # of pixels)
        cellData[j+1] = cellBorders[i+1];    // pos/neg (1/0) from segmentation
        cellData[j+2] = cellBorders[i+2];    // max marker value in cell
        cellData[j+3] = k;                   // begin index in outlines array
        for (const idx of cellBorders[i+3])
            cellOutlines[k++] = idx;
        cellData[j+4] = k;                   // end (1 past last) index in outlines array
    }

    //console.log('Total cells:', cellBorders.length/4);
    //console.log('Total cells:', cellData.length/5);

    return {posneg: posneg, cellInfo: cellData, cellOutlineIndices: cellOutlines, cellSizesSqrt: cellSizesSqrt};
}




export async function extractCells(data, mask, segThresh)
{
    let posneg0 = await createPosNegMask(data, mask, segThresh);
    posneg0 = await markBackground(data, posneg0);

    const {posneg, cellInfo, cellOutlineIndices, cellSizesSqrt} = await computeCellMapping(data, posneg0);

    let maxOD = 0;
    for (let i = 0; i < cellInfo.length; i += 5)
        if (cellInfo[i+2] > maxOD)
            maxOD = cellInfo[i+2];

    return {posneg: posneg, cellData: {cellInfo: cellInfo, cellOutlineIndices: cellOutlineIndices, cellSizesSqrt: cellSizesSqrt}, maxOD: maxOD};
}



export async function createOverlayMask(cellData, width, height, sizeRange, markerRange)
{
    const mask = new Int8Array(new Buffer(width*height));
    const cellInfo = cellData.cellInfo;
    const cellOutlines = cellData.cellOutlineIndices;
    const score = {countTotal: 0, countPos: 0};

    for (let i = 0; i < cellInfo.length; i += 5) {
        if (cellInfo[i] > sizeRange[0] && cellInfo[i] < sizeRange[1]) {
            let val = (cellInfo[i+1] == 1) ? 1 : -1;
            if (cellInfo[i+2] > markerRange[1])
                val = -2;
            else if (cellInfo[i+2] < markerRange[0])
                val = -2;
            for (let j = cellInfo[i+3]; j < cellInfo[i+4]; ++j)
                mask[cellOutlines[j]] = val;
            ++score.countTotal;
            if (val > 0)
                ++score.countPos;
        }
    }

    score.percentPos = (score.countTotal > 0) ? Math.round(1000 * score.countPos / score.countTotal) / 10 : 0;
    const extra = (Math.floor(score.percentPos) == score.percentPos) ? '.0' : '';
    score.percentPosStr = score.percentPos + extra + ' %';
    
    return {overlay: mask, score: score};
}