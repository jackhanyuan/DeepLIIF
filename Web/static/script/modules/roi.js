import {Buffer, bresenham} from './helper.js';


export const regions = [];
export let mask = null;


export function initDrawing(canvasElement, imageData, mapCanvasToImageCoordFunc, updateViewFunc)
{
    canvas = canvasElement;
    imageWidth = imageData.width;
    imageHeight = imageData.height;

    mapCoords = mapCanvasToImageCoordFunc;
    updateView = updateViewFunc;

    document.body.addEventListener('pointerdown', (e) => { pointerIds.add(e.pointerId); }, true);
    document.body.addEventListener('pointerup', (e) => { pointerIds.delete(e.pointerId); }, true);
    document.body.addEventListener('pointercancel', (e) => { pointerIds.delete(e.pointerId); }, true);
    canvasElement.addEventListener('touchstart', touchStart, {passive: false});
    canvasElement.addEventListener('touchmove', touchMove, {passive: false});
    canvasElement.addEventListener('pointerdown', pointerDown);
    document.body.addEventListener('pointermove', pointerMove);
    document.body.addEventListener('pointerup', pointerUp);
    document.body.addEventListener('pointercancel', cancelDrawing);
    document.body.addEventListener('pointerleave', cancelDrawing);

    mask = new Uint8Array(new Buffer(imageWidth * imageHeight));
    mask.fill(255);
}


export function disableDrawMode()
{
    drawMode = null;
}


export function setDrawModeSelect()
{
    drawMode = 'select';
}


export function setDrawModeExclude()
{
    drawMode = 'exclude';
}


export function drawModeActive()
{
    return (drawMode !== null);
}


export function numRegions()
{
    let count = 0;
    for (const region of regions)
        if (region.active !== null)
            ++count;
    return count;
}


export function deleteAllRegions()
{
    while (regions.length > 0)
        regions.pop();
    createGlobalMask();
}


export function deleteRegion(index)
{
    if (index >= 0 && index < regions.length)
        regions.splice(index, 1);
    createGlobalMask();
}


export function setRegionState(index, active)
{
    if (index >= 0 && index < regions.length)
        regions[index].active = active;
    createGlobalMask();
}


export function showAllRegions()
{
    for (const region of regions)
        region.active = true;
    createGlobalMask();
}


export function showNoRegions()
{
    for (const region of regions)
        region.active = false;
    createGlobalMask();
}


export function showOnlyRegion(index)
{
    if (index >= 0 && index < regions.length) {
        for (const region of regions)
            region.active = false;
        regions[index].active = true;
    }
    createGlobalMask();
}


export function drawOutlinesOnCopy(imageData, lineWidth)
{
    const outlines = new ImageData(imageData.width, imageData.height);
    outlines.data.set(imageData.data);
    const lineHalfWidth = Math.round((lineWidth - 1) / 2);
    for (const region of regions) {
        if (region.active === null || region.active) {
            const color = region.select ? [255, 255, 0] : [255, 0, 255];
            let prev = region.active === null ? region.points[0] : region.points[region.points.length-1];
            for (const p of region.points) {
                let line = bresenham(prev[0], prev[1], p[0], p[1]);
                for (let i = 0; i < line.length; i += 2) {
                    for (let dx = -lineHalfWidth; dx <= lineHalfWidth; ++dx)
                        for (let dy = -lineHalfWidth; dy <= lineHalfWidth; ++dy) {
                            const x = line[i] + dx;
                            const y = line[i+1] + dy;
                            if (x >= 0 && x < outlines.width && y >= 0 && y < outlines.height) {
                                const idx = (y * outlines.width + x) * 4;
                                outlines.data[idx] = color[0];
                                outlines.data[idx+1] = color[1];
                                outlines.data[idx+2] = color[2];
                            }
                        }
                }
                prev = p;
            }
        }
    }
    return outlines;
}


export function loadRegions(rois)
{
    loading = true;
    regions.length = 0;
    mask.fill(255);
    const rs = rois.regions;
    for (const r of rs) {
        regions.push({
            id: r.id,
            points: r.points,
            select: r.mode === 'select',
            active: r.visible,
            name: r.displayName,
        });
        createRegionMask();
    }
    counter = rois.counter;
    loading = false;
    createGlobalMask();
}


export function saveRegions()
{
    const rs = [];
    for (const region of regions) {
        if (region.active !== null) {
            const r = {};
            r['id'] = region.id;
            r['points'] = region.points;
            r['mode'] = region.select ? 'select' : 'exclude';
            r['visible'] = region.active;
            r['displayName'] = region.name;
            rs.push(r);
        }
    }
    const rois = {regions: rs, counter: counter};
    return rois;
}


let canvas = null;
let imageWidth = 0, imageHeight = 0;
let mapCoords = null;
let updateView = null;

let drawMode = null;
let drawing = false;
const pointerIds = new Set();
let counter = 0;
let loading = false;


const touch = {startX: null, startY: null, preventDefault: false};


function touchStart(e)
{
    if (e.touches.length === 1 && drawMode !== null) {
        touch.startX = e.touches[0].screenX;
        touch.startY = e.touches[0].screenY;
        touch.preventDefault = false;
    }
}


function touchMove(e)
{
    if (e.touches.length === 1 && drawMode !== null) {
        if (touch.startX !== null) {
            const diffX = Math.abs(touch.startX - e.touches[0].screenX);
            const diffY = Math.abs(touch.startY - e.touches[0].screenY);
            touch.startX = null;
            touch.startY = null;
            if (diffX >= diffY)
                touch.preventDefault = true;
        }
        if (touch.preventDefault)
            e.preventDefault();
    }
}


function pointerDown(e)
{
    if (drawMode !== null && e.buttons === 1 && pointerIds.size === 1 && !e.ctrlKey && !e.shiftKey) {
        drawing = true;
        const {x, y} = mapCoords(e.offsetX, e.offsetY);
        regions.push({
            points: [[x, y]],
            select: drawMode === 'select',
            active: null,
        });
    }
    else if (drawing && pointerIds.size > 1)
        cancelDrawing();
}


function pointerMove(e)
{
    if (e.buttons === 1 && pointerIds.size === 1 && drawing) {
        const rect = canvas.getBoundingClientRect();
        const {x, y} = mapCoords(e.clientX - rect.left - 1, e.clientY - rect.top - 1);
        regions[regions.length-1].points.push([x, y]);
        updateView();
    }
}


function pointerUp(e)
{
    if (drawing) {
        if (e.target.id === 'result-canvas' && regions[regions.length-1].points.length > 3) {
            //regions[regions.length-1].points.push(regions[regions.length-1].points[0]);
            drawing = false;
            createRegionMask();
        }
        else
            cancelDrawing();
    }
}


function cancelDrawing()
{
    if (drawing) {
        regions.pop();
        drawing = false;
        updateView();
    }
}


function createRegionMask()
{
    const latestMask = new Uint8Array(new Buffer(imageWidth * imageHeight));
    latestMask.fill(128);
    const contour = regions[regions.length-1];
    const val = contour.select ? 255 : 0;
    let prev = contour.points[contour.points.length-1];
    for (const p of contour.points) {
        let line = bresenham(prev[0], prev[1], p[0], p[1]);
        for (let i = 0; i < line.length; i += 2)
            latestMask[line[i+1]*imageWidth + line[i]] = val;
        prev = p;
    }

    const border = [];
    for (let x = 0; x < imageWidth; ++x) {
        border.push(x);
        border.push(0);
        border.push(x);
        border.push(imageHeight-1);
    }
    for (let y = 1; y < imageHeight-1; ++y) {
        border.push(0);
        border.push(y);
        border.push(imageWidth-1);
        border.push(y);
    }

    region_grow(latestMask, imageWidth, imageHeight, border, 128, 127);

    for (let i = 0; i < latestMask.length; ++i)
        if (latestMask[i] == 128)
            latestMask[i] = val;

    regions[regions.length-1].mask = latestMask;
    if (!loading)
        regions[regions.length-1].active = true;

    let minX = contour.points[0][0], minY = contour.points[0][1];
    let maxX = minX, maxY = minY;
    for (const p of contour.points) {
        if (p[0] < minX)
            minX = p[0];
        else if (p[0] > maxX)
            maxX = p[0];
        if (p[1] < minY)
            minY = p[1];
        else if (p[1] > maxY)
            maxY = p[1];
    }
    if (!loading) {
        ++counter;
        regions[regions.length-1].id = counter;
        regions[regions.length-1].name = 'Region ' + counter + ' (' + minX + ',' + minY + ')-(' + maxX + ',' + maxY + ')';
        createGlobalMask();
    }
}


function createGlobalMask()
{
    mask.fill(255);
    let isFirstSelect = true;
    for (const region of regions) {
        if (region.active) {
            if (region.select && isFirstSelect) {
                mask.fill(0);
                isFirstSelect = false;
            }
            for (let i = 0; i < mask.length; ++i)
                if (region.mask[i] === 0 || region.mask[i] === 255)
                    mask[i] = region.mask[i];
        }
    }
    updateView(true);
}


function region_grow(pixels, width, height, seeds, target, fill)
{
    const neigh_x = [0, -1, 1, 0];
    const neigh_y = [-1, 0, 0, 1];
    let pixels_x = [];
    let pixels_y = [];

    for (let i = 0; i < seeds.length; i += 2) {
        const idx = seeds[i+1]*width + seeds[i];
        if (pixels[idx] == target) {
            pixels_x.push(seeds[i]);
            pixels_y.push(seeds[i+1]);
            pixels[idx] = fill;
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
                if (pixels[ni] == target) {
                    pixels_x.push(nx);
                    pixels_y.push(ny);
                    pixels[ni] = fill;
                }
            }
        }
    }
}