export let shiftX = 0, shiftY = 0;
export let zoom = 0;


export function initPanZoom(canvasElement, imageData, updateViewFunc)
{
    canvas = canvasElement;
    imageWidth = imageData.width;
    imageHeight = imageData.height;
    heightRatio = imageHeight / imageWidth;
    calculateZoom();
    updateView = updateViewFunc;

    window.addEventListener('resize', calculateZoom);
    document.body.addEventListener('contextmenu', (e) => { e.preventDefault(); }, {passive: false});
    document.body.addEventListener('wheel', (e) => { if (e.ctrlKey) e.preventDefault(); }, {passive: false});
    canvas.addEventListener('mousedown', (e) => { e.preventDefault(); }, {passive: false});
    canvas.addEventListener('touchstart', (e) => { if (e.touches.length > 1) e.preventDefault(); }, {passive: false});
    canvas.addEventListener('pointerdown', pointerDown);
    canvas.addEventListener('pointermove', pointerMove);
    canvas.addEventListener('pointerup', pointerUp);
    canvas.addEventListener('pointercancel', pointerCancel);
    canvas.addEventListener('wheel', wheel);
}


export function zoomDisplay()
{
    return Math.round(zoom * 1000) / 10;
}


let canvas = null;
let imageWidth = 0, imageHeight = 0;
let heightRatio = 0;
let updateView = null;


const pointers = {
    cache: [],
    addCachedEvent: function(e) {
        this.removeCachedEvent(e);
        this.cache.push({id: e.pointerId, currX: e.offsetX, currY: e.offsetY, updated: false});
    },
    updateCachedEvent: function(e) {
        const index = this.cache.findIndex((element) => element.id === e.pointerId);
        if (index >= 0) {
            this.cache[index].prevX = this.cache[index].currX;
            this.cache[index].prevY = this.cache[index].currY;
            this.cache[index].currX = e.offsetX;
            this.cache[index].currY = e.offsetY;
            this.cache[index].updated = true;
        }
    },
    removeCachedEvent: function(e) {
        const index = this.cache.findIndex((element) => element.id === e.pointerId);
        if (index >= 0)
            this.cache.splice(index, 1);
    },
    count: function() {
        return this.cache.length;
    },
    resetUpdated: function() {
        for (const element of this.cache)
            element.updated = false;
    },
    allUpdated: function() {
        for (const element of this.cache)
            if (!element.updated)
                return false;
        return true;
    },
    currentCenter: function() {
        if (this.count() === 2)
            return {x: (this.cache[0].currX + this.cache[1].currX) / 2, y: (this.cache[0].currY + this.cache[1].currY) / 2};
        else
            return null;
    },
    previousCenter: function() {
        if (this.count() === 2)
            return {x: (this.cache[0].prevX + this.cache[1].prevX) / 2, y: (this.cache[0].prevY + this.cache[1].prevY) / 2};
        else
            return null;
    },
    currentDistance: function() {
        if (this.count() === 2)
            return Math.hypot(this.cache[0].currX - this.cache[1].currX, this.cache[0].currY - this.cache[1].currY);
        else
            return null;
    },
    previousDistance: function() {
        if (this.count() === 2)
            return Math.hypot(this.cache[0].prevX - this.cache[1].prevX, this.cache[0].prevY - this.cache[1].prevY);
        else
            return null;
    },
};


function pointerDown(e)
{
    pointers.addCachedEvent(e);
    if (pointers.count() === 2) {
        pointers.resetUpdated();
    }
}


function pointerMove(e)
{
    pointers.updateCachedEvent(e);

    let doZoom = false;
    let doPan = false;
    let zoomFactor = 0;
    let panX = 0, panY = 0;

    if (e.pointerType === 'mouse' && pointers.count() === 1)
    {
        if (e.buttons === 2 || (e.buttons === 1 && e.ctrlKey && !e.shiftKey)) { // right button OR ctrl + left button
            doZoom = true;
            zoomFactor = 10 * (pointers.cache[0].prevY - pointers.cache[0].currY);
        }
        else if (e.buttons === 4 || (e.buttons === 1 && e.shiftKey && !e.ctrlKey)) { // middle button OR shift + left button
            doPan = true;
            panX = e.offsetX - pointers.cache[0].prevX;
            panY = e.offsetY - pointers.cache[0].prevY;
        }
    }
    else if (e.pointerType === 'touch' && pointers.count() === 2 && pointers.allUpdated())
    {
        zoomFactor = 10 * (pointers.currentDistance() - pointers.previousDistance());
        if (Math.abs(zoomFactor) > 5)
            doZoom = true;
        const dirX0 = (pointers.cache[0].currX - pointers.cache[0].prevX) > 0;
        const dirY0 = (pointers.cache[0].currY - pointers.cache[0].prevY) > 0;
        const dirX1 = (pointers.cache[1].currX - pointers.cache[1].prevX) > 0;
        const dirY1 = (pointers.cache[1].currY - pointers.cache[1].prevY) > 0;
        const dX0 = (pointers.cache[0].currX - pointers.cache[0].prevX);
        const dY0 = (pointers.cache[0].currY - pointers.cache[0].prevY);
        const dX1 = (pointers.cache[1].currX - pointers.cache[1].prevX);
        const dY1 = (pointers.cache[1].currY - pointers.cache[1].prevY);
        if (dirX0 === dirX1 && dirY0 === dirY1) {
            doZoom = false;
            doPan = true;
            const currentCenter = pointers.currentCenter();
            const previousCenter = pointers.previousCenter();
            panX = currentCenter.x - previousCenter.x;
            panY = currentCenter.y - previousCenter.y;
        }
        pointers.resetUpdated();
    }

    if (doZoom)
    {
        applyZoomFactor(zoomFactor);
    }
    if (doPan)
    {
        shiftX += (1 / zoom) * panX;
        shiftY += (1 / zoom) * panY;
    }

    if (doZoom || doPan)
    {
        keepImageInView();
        updateView();
    }
}


function pointerUp(e)
{
    pointers.removeCachedEvent(e);
}


function pointerCancel(e)
{
    pointers.removeCachedEvent(e);
}


function wheel(e)
{
    if (e.ctrlKey)
    {
        let zoomFactor = -e.deltaY;
        if (Math.abs(zoomFactor) < 10)
            zoomFactor *= 10;

        applyZoomFactor(zoomFactor);
        keepImageInView();
        updateView();
    }
}


function applyZoomFactor(zoomFactor)
{
    const midX = -shiftX + (canvas.width / 2);
    const midY = -shiftY + (canvas.height / 2);

    let width = canvas.width - zoomFactor;
    if (width < canvas.clientWidth)
        width = canvas.clientWidth;
    if (width > imageWidth)
        width = imageWidth;
    canvas.width = width;
    canvas.height = width * heightRatio;

    shiftX = (canvas.width / 2) - midX;
    shiftY = (canvas.height / 2) - midY;

    calculateZoom();
}


function calculateZoom()
{
    zoom = canvas.clientWidth / canvas.width;
}


function keepImageInView()
{
    if (shiftX > 0)
        shiftX = 0;
    if (shiftY > 0)
        shiftY = 0;
    if (shiftX + imageWidth < canvas.width)
        shiftX = canvas.width - imageWidth;
    if (shiftY + imageHeight < canvas.height)
        shiftY = canvas.height - imageHeight;
}