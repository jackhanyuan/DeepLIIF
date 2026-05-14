export const useShared = (window.hasOwnProperty('crossOriginIsolated') && crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined');
export const Buffer = (useShared) ? SharedArrayBuffer : ArrayBuffer;
//console.log('useShared =', useShared, ' | ', Buffer);


export function getImageData(img)
{
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}


export function imageToCanvas(img)
{
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext('2d').putImageData(img, 0, 0);
    return canvas;
}


export function imageToBlob(img)
{
    return new Promise((resolve) => {
        imageToCanvas(img).toBlob((blob) => {
            resolve(blob);
        });
    });
}


export function createWorker(workerFunc)
{
    const src = 'onmessage = ' + workerFunc.toString();
    return new Worker(URL.createObjectURL(new Blob([src])));
}


//export function toGrayscale(image)
//{
//    let gray = new Uint8Array(new Buffer(image.width*image.height));
//    for (let i = 0, j = 0; i < gray.length; ++i, j += 4)
//        gray[i] = Math.round(image.data[j]*0.299 + image.data[j+1]*0.587 + image.data[j+2]*0.114);
//    return gray;
//}


//export function toGrayscaleMax(image)
//{
//    let gray = new Uint8Array(new Buffer(image.width*image.height));
//    for (let i = 0, j = 0; i < gray.length; ++i, j += 4)
//        gray[i] = Math.max(image.data[j], image.data[j+1], image.data[j+2]);
//    return gray;
//}


export function bresenham(x0, y0, x1, y1)
{
    const dx = Math.abs(x1 - x0);
    const sx = x0 < x1 ? 1 : -1;
    const dy = -Math.abs(y1 - y0)
    const sy = y0 < y1 ? 1 : -1
    let error = dx + dy
    let points = [];
    
    while (true) {
        points.push(x0, y0);
        if (x0 == x1 && y0 == y1)
            break;
        let e2 = 2 * error;
        if (e2 >= dy) {
            if (x0 == x1)
                break;
            error = error + dy;
            x0 = x0 + sx;
        }
        if (e2 <= dx) {
            if (y0 == y1)
                break;
            error = error + dx;
            y0 = y0 + sy;
        }
    }

    return points;
}