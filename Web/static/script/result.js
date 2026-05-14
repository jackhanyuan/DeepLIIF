import {useShared, Buffer, getImageData, imageToCanvas, imageToBlob} from './modules/helper.js';
import * as pp from './modules/postprocess.js';
import {createKDE, calculateDefaultSizeThresh} from './modules/distribution.js';
import * as pz from './modules/panzoom.js';
import * as roi from './modules/roi.js';




const result = {settings: null, processing: false};

const view = {
    selectedCellSize: 0,
    maskUpdated: false,
};




function initInterface()
{
    window.addEventListener('resize', updateResults);

    //document.body.addEventListener('keydown', (e) => { console.log(e.key, typeof e.key, (e.key - 1)); });
    //document.body.addEventListener('keydown', (e) => {
    //    //roi.regions[e.key-1].active = !roi.regions[e.key-1].active;
    //    //updateResults();
    //    roi.setRegionState(e.key-1, !roi.regions[e.key-1].active);
    //});

    ui.resultCanvas = document.getElementById('result-canvas');
    ui.setResultSize = (img) => {
        ui.resultCanvas.style.maxWidth = img.width + 'px';
        ui.resultCanvas.width = img.width;
        ui.resultCanvas.height = img.height;
    };

    ui.resultCanvas.addEventListener('click', resultClick);

    ui.distributionCanvas = document.getElementById('size-distribution-canvas');
    ui.getDistributionWidth = () => ui.distributionCanvas.width;
    ui.getDistributionHeight = () => ui.distributionCanvas.height;
    ui.setDistributionSize = () => {
        if (ui.distributionCanvas.width != ui.distributionCanvas.offsetWidth) {
            //console.log('distrcanvas:', ui.distributionCanvas.offsetWidth, ui.distributionCanvas.clientWidth);
            ui.distributionCanvas.width = ui.distributionCanvas.offsetWidth;
            ui.distributionCanvas.height = ui.distributionCanvas.offsetHeight;
        }
    }

    ui.histogramCanvas = document.getElementById('marker-histogram-canvas');
    ui.getHistogramWidth = () => ui.histogramCanvas.width;
    ui.getHistogramHeight = () => ui.histogramCanvas.height;
    ui.setHistogramSize = () => {
        if (ui.histogramCanvas.width != ui.histogramCanvas.offsetWidth) {
            ui.histogramCanvas.width = ui.histogramCanvas.offsetWidth;
            ui.histogramCanvas.height = ui.histogramCanvas.offsetHeight;
        }
    }

    ui.segThreshSlider = document.getElementById('slider-prob');
    ui.segThreshSlider.addEventListener('input', updateResults);
    ui.segThreshSliderOutput = document.getElementById('slider-prob-output');
    ui.resetSegThresh = document.getElementById('reset-seg-thresh');
    ui.resetSegThresh.addEventListener('click', resetOption);
    //ui.numberSegThresh = document.getElementById('number-prob');
    //ui.numberSegThresh.addEventListener('input', (event) => {
    //    let val = ui.numberSegThresh.value;
    //    if (val == '')
    //        val = 0;
    //    ui.segThreshSlider.value = val;
    //    updateResults();
    //});
    ui.segThreshSlider.addEventListener('pointerdown', updateRangeOutput);
    ui.segThreshSlider.addEventListener('input', updateRangeOutput);
    ui.segThreshSlider.addEventListener('pointerup', updateRangeOutput);
    ui.segThreshSlider.addEventListener('pointercancel', updateRangeOutput);

    ui.sizeThreshSlider1 = document.getElementById('slider-size-min');
    ui.sizeThreshSlider1.addEventListener('input', updateResults);
    ui.sizeThreshSlider2 = document.getElementById('slider-size-max');
    ui.sizeThreshSlider2.addEventListener('input', updateResults);
    ui.resetSizeThresh = document.getElementById('reset-size-thresh');
    ui.resetSizeThresh.addEventListener('click', resetOption);
    //ui.numberSizeThreshMin = document.getElementById('number-size-min');
    //ui.numberSizeThreshMax = document.getElementById('number-size-max');
    //ui.numberSizeThreshMin.addEventListener('input', (event) => {
    //    let minVal = ui.numberSizeThreshMin.value;
    //    let maxVal = ui.numberSizeThreshMax.value;
    //    if (minVal == '')
    //        minVal = 0;
    //    if (maxVal == '')
    //        maxVal = 0;
    //    ui.sizeThreshSlider1.value = Math.sqrt(minVal);
    //    ui.sizeThreshSlider2.value = Math.sqrt(maxVal);
    //    updateResults();
    //});
    //ui.numberSizeThreshMax.addEventListener('input', (event) => {
    //    let minVal = ui.numberSizeThreshMin.value;
    //    let maxVal = ui.numberSizeThreshMax.value;
    //    if (minVal == '')
    //        minVal = 0;
    //    if (maxVal == '')
    //        maxVal = 0;
    //    ui.sizeThreshSlider1.value = Math.sqrt(minVal);
    //    ui.sizeThreshSlider2.value = Math.sqrt(maxVal);
    //    updateResults();
    //});
    ui.sizeThreshSlider1.addEventListener('pointerdown', updateRangeOutput);
    ui.sizeThreshSlider1.addEventListener('input', updateRangeOutput);
    ui.sizeThreshSlider1.addEventListener('pointerup', updateRangeOutput);
    ui.sizeThreshSlider1.addEventListener('pointercancel', updateRangeOutput);
    ui.sizeThreshSlider2.addEventListener('pointerdown', updateRangeOutput);
    ui.sizeThreshSlider2.addEventListener('input', updateRangeOutput);
    ui.sizeThreshSlider2.addEventListener('pointerup', updateRangeOutput);
    ui.sizeThreshSlider2.addEventListener('pointercancel', updateRangeOutput);

    ui.markerThreshSlider1 = document.getElementById('slider-marker-lower');
    ui.markerThreshSlider1.addEventListener('input', updateResults);
    ui.markerThreshSlider2 = document.getElementById('slider-marker-upper');
    ui.markerThreshSlider2.addEventListener('input', updateResults);
    ui.resetMarkerThresh = document.getElementById('reset-marker-thresh');
    ui.resetMarkerThresh.addEventListener('click', resetOption);
    //ui.numberMarkerThresh = document.getElementById('number-marker');
    //ui.numberMarkerThresh.addEventListener('input', (event) => {
    //    let val = ui.numberMarkerThresh.value;
    //    if (val == '')
    //        val = 0;
    //    ui.markerThreshSlider.value = val;
    //    updateResults();
    //});
    ui.markerThreshSlider1.addEventListener('pointerdown', updateRangeOutput);
    ui.markerThreshSlider1.addEventListener('input', updateRangeOutput);
    ui.markerThreshSlider1.addEventListener('pointerup', updateRangeOutput);
    ui.markerThreshSlider1.addEventListener('pointercancel', updateRangeOutput);
    ui.markerThreshSlider2.addEventListener('pointerdown', updateRangeOutput);
    ui.markerThreshSlider2.addEventListener('input', updateRangeOutput);
    ui.markerThreshSlider2.addEventListener('pointerup', updateRangeOutput);
    ui.markerThreshSlider2.addEventListener('pointercancel', updateRangeOutput);

    ui.modeShowSize = document.getElementById('modeShowSize');
    //ui.modeShowSize.addEventListener('input', setDrawMode);
    ui.modeShowSize.addEventListener('input', roi.disableDrawMode);
    ui.modeSelectRegion = document.getElementById('modeSelectRegion');
    //ui.modeSelectRegion.addEventListener('input', setDrawMode);
    ui.modeSelectRegion.addEventListener('input', roi.setDrawModeSelect);
    ui.modeExcludeRegion = document.getElementById('modeExcludeRegion');
    //ui.modeExcludeRegion.addEventListener('input', setDrawMode);
    ui.modeExcludeRegion.addEventListener('input', roi.setDrawModeExclude);
    ui.resetMask = document.getElementById('reset-mask');
    ui.resetMask.addEventListener('click', resetMask);

    ui.originalViewButton = document.getElementById('viewOriginal');
    ui.originalViewButton.addEventListener('input', updateResults);
    ui.multiplexViewButton = document.getElementById('viewMultiplex');
    ui.multiplexViewButton.addEventListener('input', updateResults);
    ui.showOverlayButton = document.getElementById('showOverlay');
    ui.showOverlayButton.addEventListener('input', updateResults);

    ui.showMarkerButton = document.getElementById('showMarker');
    ui.showMarkerButton.addEventListener('input', updateResults);
    ui.markerRangeSlider1 = document.getElementById('markerMin');
    ui.markerRangeSlider1.addEventListener('input', updateResults);
    ui.markerRangeSlider2 = document.getElementById('markerMax');
    ui.markerRangeSlider2.addEventListener('input', updateResults);
    ui.resetMarker = document.getElementById('reset-marker-range');
    ui.resetMarker.addEventListener('click', resetOption);

    ui.showDapiButton = document.getElementById('showDapi');
    ui.showDapiButton.addEventListener('input', updateResults);
    ui.dapiRangeSlider1 = document.getElementById('dapiMin');
    ui.dapiRangeSlider1.addEventListener('input', updateResults);
    ui.dapiRangeSlider2 = document.getElementById('dapiMax');
    ui.dapiRangeSlider2.addEventListener('input', updateResults);
    ui.resetDapi = document.getElementById('reset-dapi-range');
    ui.resetDapi.addEventListener('click', resetOption);

    ui.showLap2Button = document.getElementById('showLap2');
    ui.showLap2Button.addEventListener('input', updateResults);
    ui.lap2RangeSlider1 = document.getElementById('lap2Min');
    ui.lap2RangeSlider1.addEventListener('input', updateResults);
    ui.lap2RangeSlider2 = document.getElementById('lap2Max');
    ui.lap2RangeSlider2.addEventListener('input', updateResults);
    ui.resetLap2 = document.getElementById('reset-lap2-range');
    ui.resetLap2.addEventListener('click', resetOption);

    ui.downloadFiles = document.getElementById('download-button');
    ui.downloadFiles.addEventListener('click', downloadFiles);
    ui.preparingFiles = document.getElementById('preparing-files');
}


function fetchImages()
{
    for (const prop in imageNames) {
        let img = document.createElement('img');
        img.id = prop;
        img.addEventListener('load', imageLoaded);
        //img.src = "static/deepliif_results_website/" + imageNames[prop] + ".png"
        //img.src = "static/deepliif_results_4096x4096/" + imageNames[prop] + ".png"
        img.src = '/image/' + imageID + imageNames[prop];
        imageElements[prop] = img;
    }
}


function fetchSavedState()
{
    const xhr = new XMLHttpRequest();
    xhr.onload = () => {
        if (xhr.status === 200) {
            savedState = JSON.parse(xhr.responseText);
        } else {
            console.log(xhr.responseText);
            savedState = {};
        }
    };
    xhr.open('GET', '/result/state/'+imageID);
    xhr.send();
}

let savedState = null;


function initPage()
{
    initInterface();
    fetchImages();
    fetchSavedState();

    //document.getElementById('message').innerHTML = 'useShared: ' + useShared;
    //console.log('useShared:', useShared);

    //const distributionCanvas = document.getElementById('distribution-canvas');
    //distributionCanvas.width = distributionCanvas.offsetWidth;
    //distributionCanvas.height = distributionCanvas.offsetHeight;
    //var ctx = distributionCanvas.getContext("2d");
    //ctx.fillStyle = "rgb(200, 200, 200)";
    //ctx.fillRect(0, 0, distributionCanvas.width, distributionCanvas.height);
}


const ui = {};
let imageFetchRemaining = Object.keys(imageNames).length;
let imageElements = {}, imageData = {};


if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}




function imageLoaded(e)
{
    imageData[e.target.id] = getImageData(e.target);
    --imageFetchRemaining;
    //console.log('Remaining images:', imageFetchRemaining);
    if (imageFetchRemaining == 0)
        initData();
}




function imageFailed(e)
{
    --imageFetchRemaining;
    if (imageFetchRemaining == 0)
        initData();
}




function calculateDefaults()
{
    ui.defaultSegThresh = 120;//150;
    ui.defaultSizeThresh = 7;
    /* For old mIF outputs
    ui.defaultMarkerThresh = 255;
    ui.defaultMarkerRange = pp.calculateDisplayRange(pp.createHistogram(pixelData.marker));
    ui.defaultDapiRange = pp.calculateDisplayRange(pp.createHistogram(pixelData.dapi));
    ui.defaultLap2Range = pp.calculateDisplayRange(pp.createHistogram(pixelData.lap2));
    */
    //ui.defaultMarkerThresh = Math.round((ui.defaultMarkerRange[1] - ui.defaultMarkerRange[0]) * 0.9) + ui.defaultMarkerRange[0];
}


function loadSettings()
{
    //ui.defaultSegThresh = savedState.defaults.segThresh;
    ui.defaultSizeThresh = Math.sqrt(savedState.defaults.sizeRange[0]);
    /* For old mIF outputs
    ui.defaultMarkerThresh = savedState.defaults.markerThresh;
    ui.defaultMarkerRange = savedState.defaults.mpMarkerRange;
    ui.defaultDapiRange = savedState.defaults.mpDapiRange;
    ui.defaultLap2Range = savedState.defaults.mpLap2Range;
    */

    //ui.segThreshSlider.value = savedState.settings.segThresh;

    ui.sizeThreshSlider1.max = Math.sqrt(savedState.settings.sizeMax);
    ui.sizeThreshSlider2.max = Math.sqrt(savedState.settings.sizeMax);
    ui.sizeThreshSlider1.value = Math.sqrt(savedState.settings.sizeRange[0]);
    ui.sizeThreshSlider2.value = Math.sqrt(savedState.settings.sizeRange[1]);

    if (savedState.dataVersion == 2) {
        ui.markerThreshSlider1.value = savedState.settings.markerThresh[0];
        ui.markerThreshSlider2.value = savedState.settings.markerThresh[1];
    }
    else {
        ui.markerThreshSlider1.value = 0;
        ui.markerThreshSlider2.value = savedState.settings.markerThresh;
    }

    /* For old mIF outputs
    ui.showMarkerButton.checked = savedState.settings.mpMarkerVisible;
    ui.markerRangeSlider1.value = savedState.settings.mpMarkerRange[0];
    ui.markerRangeSlider2.value = savedState.settings.mpMarkerRange[1];

    ui.showDapiButton.checked = savedState.settings.mpDapiVisible;
    ui.dapiRangeSlider1.value = savedState.settings.mpDapiRange[0];
    ui.dapiRangeSlider2.value = savedState.settings.mpDapiRange[1];

    ui.showLap2Button.checked = savedState.settings.mpLap2Visible;
    ui.lap2RangeSlider1.value = savedState.settings.mpLap2Range[0];
    ui.lap2RangeSlider2.value = savedState.settings.mpLap2Range[1];
    */
}




async function initData()
{
    //if (imageFetchRemaining > 0 || 

    //let canvas = document.getElementById('canvas-result');
    //canvas.style.maxWidth = (imageData.orig.width) + 'px';
    //canvas.width = imageData.orig.width;
    //canvas.height = imageData.orig.height;
    ui.setResultSize(imageData.orig);
    ui.resultHeightRatio = imageData.orig.height / imageData.orig.width;

    let start = new Date();

    pixelData.width = imageData.orig.width;
    pixelData.height = imageData.orig.height;
    pixelData.size = pixelData.width * pixelData.height;

    /* For old mIF outputs
    let start1 = new Date();
    pixelData.marker = pp.toGrayscaleMax(imageData.marker);
    pixelData.dapi = pp.toGrayscaleMax(imageData.dapi);
    pixelData.lap2 = pp.toGrayscaleMax(imageData.lap2);
    let end1 = new Date();
    */
    //console.log('mpIFs to GrayscaleMax:', (end1-start1), 'ms');
    //console.log(ui);

    pixelData.marker = pp.calcOpticalDensity(imageData.orig);

    ({segRed: pixelData.segRed, segGreen: pixelData.segGreen, segBlue: pixelData.segBlue} = pp.extractSegChannels(imageData.orig, imageData.seg));

    pixelData.mask = new Uint8Array(new Buffer(pixelData.segRed.length));
    pixelData.mask.fill(255);

    let end = new Date();
    //console.log('initPostprocess time:', (end-start), 'ms');

    //({posneg, cellData} = await extractCells(pixelData, ui.getSegThresh()));
    //prevSegThresh = ui.getSegThresh();

    // wait until the saved state is recieved
    while (savedState === null)
        await new Promise(resolve => setTimeout(resolve, 1000));
    //console.log(savedState);

    if (Object.keys(savedState).length === 0)
    {
        calculateDefaults();
        await updateResults();

        let defaultSizeBounds = [4, 7, 10]; // [min, default, max] for defaultSizeThresh at 40x
        if (inferResolution === 20)
            defaultSizeBounds = [3, 4, 6];
        else if (inferResolution === 10)
            defaultSizeBounds = [2, 2, 3];

        const sizeThresh = await calculateDefaultSizeThresh(result.cellData.cellSizesSqrt);
        if (sizeThresh >= defaultSizeBounds[0] && sizeThresh <= defaultSizeBounds[2])
            ui.defaultSizeThresh = sizeThresh;
        else if (sizeThresh < defaultSizeBounds[0])
            ui.defaultSizeThresh = defaultSizeBounds[0];
        else
            ui.defaultSizeThresh = defaultSizeBounds[1];

        ui.defaultSizeMax = ui.sizeThreshSlider1.max;
        resetOption('all');
    }
    else
    {
        loadSettings();
    }


    await updateResults();

    // ready to display to the user
    document.getElementById('processing').style.display = "none";
    document.getElementById('result-content').style.display = "block";
    await updateResults(); // get distribution to show

    pz.initPanZoom(ui.resultCanvas, imageData.orig, updateResults);
    roi.initDrawing(ui.resultCanvas, imageData.orig, mapCanvasToImageCoord, updateResults);

    if (Object.keys(savedState).length !== 0) {
        document.getElementById('result-content').style.display = "none";
        document.getElementById('processing').style.display = "block";
        if (savedState.rois.regions.length > 1)
            await new Promise(resolve => setTimeout(resolve, 100));
        roi.loadRegions(savedState.rois);
        document.getElementById('processing').style.display = "none";
        document.getElementById('result-content').style.display = "block";
    }
}


let pixelData = {};




function getCurrentSettings()
{
    return {
        segThresh: parseInt(ui.segThreshSlider.value),
        //sizeThresh: parseInt(ui.sizeThreshSlider1.value ** 2),
        sizeRange: [Math.round(parseFloat(ui.sizeThreshSlider1.value ** 2)),
                    Math.round(parseFloat(ui.sizeThreshSlider2.value ** 2))].sort((a,b)=>a-b),
        markerThresh: [parseInt(ui.markerThreshSlider1.value),
                       parseInt(ui.markerThreshSlider2.value)].sort((a,b)=>a-b),
        originalView: ui.originalViewButton.checked,
        showOverlay: ui.showOverlayButton.checked,
        showMarker: ui.showMarkerButton.checked,
        markerRange: [parseInt(ui.markerRangeSlider1.value),
                      parseInt(ui.markerRangeSlider2.value)].sort((a,b)=>a-b),
        showDapi: ui.showDapiButton.checked,
        dapiRange: [parseInt(ui.dapiRangeSlider1.value),
                    parseInt(ui.dapiRangeSlider2.value)].sort((a,b)=>a-b),
        showLap2: ui.showLap2Button.checked,
        lap2Range: [parseInt(ui.lap2RangeSlider1.value),
                    parseInt(ui.lap2RangeSlider2.value)].sort((a,b)=>a-b),
    };
}


function isSettingChanged(settings, prop = null)
{
    if (settings == null && result.settings == null)
        return false;
    if (settings == null || result.settings == null)
        return true;

    if (prop == null) {
        for (const p in settings)
            if (isSettingChanged(settings, p))
                return true;
    }
    else if (Array.isArray(prop)) {
        for (const p of prop)
            if (isSettingChanged(settings, p))
                return true;
    }
    else if (Array.isArray(settings[prop])) {
        for (let i = 0; i < settings[prop].length; ++i)
            if (settings[prop][i] != result.settings[prop][i])
                return true;
    }
    else {
        if (settings[prop] != result.settings[prop])
            return true;
    }

    return false;
}


function resetOption(e)
{
    //console.log('resetOption:', e);
    if (e != 'all')
        e.preventDefault();

    if (e == 'all' || e.target.id == 'reset-seg-thresh') {
        ui.segThreshSlider.value = ui.defaultSegThresh;
    }
    if (e == 'all' || e.target.id == 'reset-size-thresh') {
        ui.sizeThreshSlider1.value = ui.defaultSizeThresh;
        ui.sizeThreshSlider2.value = ui.sizeThreshSlider2.max;
    }
    if (e == 'all' || e.target.id == 'reset-marker-thresh') {
        ui.markerThreshSlider1.value = 0;
        ui.markerThreshSlider2.value = 255;//ui.defaultMarkerThresh;
    }
    /* For old mIF outputs
    if (e == 'all' || e.target.id == 'reset-marker-range') {
        ui.showMarkerButton.checked = true;
        ui.markerRangeSlider1.value = ui.defaultMarkerRange[0];
        ui.markerRangeSlider2.value = ui.defaultMarkerRange[1];
    }
    if (e == 'all' || e.target.id == 'reset-dapi-range') {
        ui.showDapiButton.checked = true;
        ui.dapiRangeSlider1.value = ui.defaultDapiRange[0];
        ui.dapiRangeSlider2.value = ui.defaultDapiRange[1];
    }
    if (e == 'all' || e.target.id == 'reset-lap2-range') {
        ui.showLap2Button.checked = true;
        ui.lap2RangeSlider1.value = ui.defaultLap2Range[0];
        ui.lap2RangeSlider2.value = ui.defaultLap2Range[1];
    }
    */

    updateResults();
}


function resultClick(e)
{
    //console.log('click');
    if (!roi.drawModeActive() && e.button == 0 && !e.ctrlKey && !e.shiftKey) {
        //console.log('show size marker');
        selectCellSize(e.offsetX, e.offsetY);
    }
}



function updateRangeOutput(e)
{
    const output = document.getElementById(e.target.id+'-output');
    if (e.type == 'pointerdown' || e.type == 'input') {
        const value = e.target.value;
        if (e.target.id.includes('size'))
            output.innerHTML = Math.round(value ** 2);
        else
            output.innerHTML = value;
        const min = e.target.min;
        const max = e.target.max;
        const percentage = (value - min) * 100 / (max - min);
        output.style.right = 'auto';
        output.style.left = percentage + '%';
        output.style.transform = 'translateX(-'+percentage+'%)';
        output.style.display = 'initial';
    }
    else {
        output.style.display = 'none';
    }
}



function createRoiList()
{
    const roiContainer = document.getElementById('roi-container');
    if (roiContainer.childElementCount - 1 !== roi.numRegions())
    {
        while (roiContainer.firstChild)
            roiContainer.removeChild(roiContainer.lastChild);

        const roiHeader = document.createElement('div');

        const roiOnlyPlaceholder = document.createElement('a');
        roiOnlyPlaceholder.href = '#';
        roiOnlyPlaceholder.className = 'roi-only';
        roiOnlyPlaceholder.innerHTML = 'only';
        roiHeader.appendChild(roiOnlyPlaceholder);

        const roiAllCheckbox = document.createElement('input');
        roiAllCheckbox.type = 'checkbox';
        roiAllCheckbox.id = 'roiAll';
        roiAllCheckbox.checked = true;
        roiAllCheckbox.addEventListener('input', (e) => {
            const selectors = document.getElementsByClassName('roi-selector');
            for (const selector of selectors)
                selector.checked = e.target.checked;
            if (e.target.checked)
                roi.showAllRegions();
            else
                roi.showNoRegions();
        });
        if (roi.numRegions() === 0)
            roiAllCheckbox.style.visibility = 'hidden';
        roiHeader.appendChild(roiAllCheckbox);

        const roiAllLabel = document.createElement('label');
        roiAllLabel.setAttribute('for', roiAllCheckbox.id);
        roiAllLabel.appendChild(document.createTextNode((roi.numRegions() === 0) ? 'No ROIs' : 'All ROIs'));
        roiHeader.appendChild(roiAllLabel);

        roiContainer.appendChild(roiHeader);

        for (const [index, region] of roi.regions.entries())
        {
            if (region.active !== null)
            {
                if (!region.active)
                    roiAllCheckbox.checked = false;
                const roiDiv = document.createElement('div');

                const roiOnlyLink = document.createElement('a');
                roiOnlyLink.href = '#';
                roiOnlyLink.className = 'roi-only';
                roiOnlyLink.innerHTML = 'only';
                roiOnlyLink.addEventListener('click', (e) => {
                    e.preventDefault();
                    const selectors = document.getElementsByClassName('roi-selector');
                    for (const selector of selectors)
                        selector.checked = false;
                    if (selectors.length === 1)
                        roiAllCheckbox.checked = true;
                    else
                        roiAllCheckbox.checked = false;
                    document.getElementById('roi'+index).checked = true;
                    roi.showOnlyRegion(index);
                });
                roiDiv.appendChild(roiOnlyLink);

                const roiCheckbox = document.createElement('input');
                roiCheckbox.type = 'checkbox';
                roiCheckbox.id = 'roi' + index;
                roiCheckbox.checked = region.active;
                roiCheckbox.addEventListener('input', (e) => {
                    roiAllCheckbox.checked = true;
                    const selectors = document.getElementsByClassName('roi-selector');
                    for (const selector of selectors)
                        if (!selector.checked)
                            roiAllCheckbox.checked = false;
                    roi.setRegionState(index, e.target.checked);
                });
                roiCheckbox.classList.add('roi-selector');
                roiDiv.appendChild(roiCheckbox);

                const roiLabel = document.createElement('label');
                roiLabel.setAttribute('for', roiCheckbox.id);
                roiLabel.appendChild(document.createTextNode(region.name));
                roiDiv.appendChild(roiLabel);

                const roiDeleteLink = document.createElement('a');
                roiDeleteLink.href = '#';
                roiDeleteLink.className = 'roi-delete';
                roiDeleteLink.innerHTML = '&#x2573;';
                roiDeleteLink.addEventListener('click', (e) => {
                    e.preventDefault();
                    roi.deleteRegion(index);
                });
                roiDiv.appendChild(roiDeleteLink);

                roiDiv.className = (region.select) ? 'select' : 'exclude';
                roiContainer.appendChild(roiDiv);
            }
        }
    }
}




var needToSaveState = false;
var saveStateTimerId = 0;

async function updateResults(e)
{
    clearTimeout(saveStateTimerId);
    needToSaveState = true;

    const settings = getCurrentSettings();
    //ui.numberSegThresh.value = settings.segThresh;
    //ui.numberSizeThreshMin.value = settings.sizeRange[0];
    //ui.numberSizeThreshMax.value = settings.sizeRange[1];
    //ui.numberMarkerThresh.value = settings.markerThresh;

    createRoiList();

    if (result.processing)
        return;
    result.processing = true;

    const startTime = new Date();

    //const settings = getCurrentSettings();
    //console.log('settings:', settings);
    const maskUpdated = (e === true);// view.maskUpdated;
    //view.maskUpdated = false;

    if (settings.originalView)
        document.getElementById('multiplex-cover').style.display = 'block';
    else
        document.getElementById('multiplex-cover').style.display = 'none';

    let overlayChanged = false;
    if (isSettingChanged(settings, ['segThresh', 'sizeRange', 'markerThresh']) || maskUpdated)
    {
        overlayChanged = true;

        if (isSettingChanged(settings, 'segThresh') || maskUpdated)
        {
            //console.log('EXTRACT CELLS');
            //({posneg: result.posneg, cellData: result.cellData} = await pp.extractCells(pixelData, roi.mask, settings.segThresh));
            let maxOD = 255;
            ({posneg: result.posneg, cellData: result.cellData, maxOD: maxOD} = await pp.extractCells(pixelData, roi.mask, 120));
            ui.sizeThreshSlider1.max = Math.max.apply(null, result.cellData.cellSizesSqrt) + 1;
            ui.sizeThreshSlider2.max = ui.sizeThreshSlider1.max;

            if (ui.markerThreshSlider1.value > maxOD)
                ui.markerThreshSlider1.value = maxOD;
            if (ui.markerThreshSlider2.value > maxOD)
                ui.markerThreshSlider2.value = maxOD;
            ui.markerThreshSlider1.max = maxOD;
            ui.markerThreshSlider2.max = maxOD;

            //result.markerHistogram = new Uint32Array(256);
            //for (let i = 0; i < pixelData.marker.length; ++i)
            //    if (result.posneg[i] == 2 || result.posneg[i] == -2)
            //        ++result.markerHistogram[pixelData.marker[i]];
        }

        //console.log('PERFORM SCORING');
        ({overlay: result.overlay, score: result.score} =
            await pp.createOverlayMask(result.cellData, imageData.orig.width, imageData.orig.height, settings.sizeRange, settings.markerThresh));
    }

    ui.setDistributionSize();
    if (isSettingChanged(settings, 'segThresh') || maskUpdated || result.kde.length != ui.getDistributionWidth())
        result.kde = await createKDE(result.cellData.cellSizesSqrt, ui.getDistributionWidth(), 1.0);
    //console.log('result.kde.length =', result.kde.length);

    //ui.setHistogramSize();
    //if (typeof result.kdeHistogram === 'undefined' || result.kdeHistogram.length != ui.getHistogramWidth()) {
    //    //console.log('need to update marker histogram');
    //    //console.log(pixelData.marker);
    //    result.kdeHistogram = await createKDE(pixelData.marker, ui.getHistogramWidth(), 0.01);
    //    //console.log(result.kdeHistogram)
    //}

    /* For old mIF outputs
    let multiplexChanged = false;
    if (isSettingChanged(settings, ['showMarker', 'markerRange', 'showDapi', 'dapiRange', 'showLap2', 'lap2Range'])) {
        multiplexChanged = true;
        result.multiplex = createMultiplex(settings);
    }
    */

    // create composite image
    //if (overlayChanged || multiplexChanged || isSettingChanged(settings, ['originalView', 'showOverlay']))
    if (overlayChanged || isSettingChanged(settings, ['originalView', 'showOverlay']))
    {
        //console.log('need to update composite');
        //const bkgnd = (settings.originalView) ? imageData.orig : result.multiplex;
        const bkgnd = imageData.orig;
        const composite = new ImageData(bkgnd.width, bkgnd.height);
        composite.data.set(bkgnd.data);
        if (settings.showOverlay) {
            const overlay = result.overlay;
            for (let i = 0, j = 0; i < overlay.length; ++i, j += 4) {
                if (overlay[i] > 0) {
                    composite.data[j] = 255;
                    composite.data[j+1] = 0;
                    composite.data[j+2] = 0;
                } else if (overlay[i] < 0) {
                    composite.data[j] = 0;
                    composite.data[j+1] = 0;
                    composite.data[j+2] = 255;
                }
            }
        }
        result.composite = composite;
    }

    //const bkgnd = (settings.originalView) ? imageData.orig : result.multiplex;
    //const overlay = (settings.showOverlay) ? result.overlay : null;
    //drawResultImage(bkgnd, overlay);

    result.compositeContours = roi.drawOutlinesOnCopy(result.composite, 3/pz.zoom);

    drawResultImage();

    // move compositing to separate function?
    // save composite (bkgnd + overlay) in result.final (to use later when creating zip)?

    drawCellSizeDistribution(settings.sizeRange);
    //drawMarkerHistogram();
    updateMarkerTicks(settings.markerRange);

    //console.log(result.score);
    // { countTotal: 117, countPos: 61, percentPos: 52.1, percentPosStr: "52.1 %" }
    updateScore();

    const endTime = new Date();
    //console.log('Time:', (endTime-startTime), 'ms');

    result.settings = settings;
    result.processing = false;

    if (isSettingChanged(getCurrentSettings()))
        updateResults(null);
    else
        saveStateTimerId = setTimeout(saveState, 1000);
}


//function updateBackgroundImage(e)


/*
function drawResultImageOLD(bkgnd, overlay)
{
    const testImage = new ImageData(bkgnd.width, bkgnd.height);
    testImage.data.set(bkgnd.data);

    if (overlay != null) {
        for (let i = 0, j = 0; i < overlay.length; ++i, j += 4) {
            if (overlay[i] > 0) {
                testImage.data[j] = 255;
                testImage.data[j+1] = 0;
                testImage.data[j+2] = 0;
            } else if (overlay[i] < 0) {
                testImage.data[j] = 0;
                testImage.data[j+1] = 0;
                testImage.data[j+2] = 255;
            }
        }
    }

    ui.resultCanvas.getContext('2d').putImageData(testImage, 0, 0);
    //console.log('resultCanvas:', ui.resultCanvas.offsetWidth, ui.resultCanvas.offsetHeight);
    //console.log('resultCanvas:', ui.resultCanvas.clientWidth, ui.resultCanvas.clientHeight);
}
*/


function drawResultImage()
{
    const ctx = ui.resultCanvas.getContext('2d');
    ctx.clearRect(0, 0, ui.resultCanvas.width, ui.resultCanvas.height);
    ctx.putImageData(result.compositeContours, pz.shiftX, pz.shiftY);

    // START DEBUG - draw mask
    //const mask = new ImageData(pixelData.width, pixelData.height);
    //for (let i = 0, j = 0; i < pixelData.mask.length; ++i, j += 4) {
    //    mask.data[j] = pixelData.mask[i];
    //    mask.data[j+1] = pixelData.mask[i];
    //    mask.data[j+2] = pixelData.mask[i];
    //    mask.data[j+3] = 255;
    //}
    //ctx.putImageData(mask, pz.shiftX, pz.shiftY);
    // END DEBUG

    // START DEBUG - draw posneg
    //const mask = new ImageData(pixelData.width, pixelData.height);
    //for (let i = 0, j = 0; i < result.posneg.length; ++i, j += 4) {
    //    if (result.posneg[i] == 2) {
    //        mask.data[j] = 255;
    //        mask.data[j+1] = 0;
    //        mask.data[j+2] = 0;
    //    }
    //    else if (result.posneg[i] == -2) {
    //        mask.data[j] = 0;
    //        mask.data[j+1] = 0;
    //        mask.data[j+2] = 255;
    //    }
    //    mask.data[j+3] = 255;
    //}
    //ctx.putImageData(mask, pz.shiftX, pz.shiftY);
    //// (76, 169)
    //const idx = 169*pixelData.width + 76;
    //console.log('posneg value:', result.posneg[idx]);
    //console.log(pixelData.segRed[idx], pixelData.segGreen[idx], pixelData.segBlue[idx]);
    // END DEBUG
}


function drawCellSizeDistribution(sizeRange = [-1])
{
    const kde = result.kde;
    const kdeMax = Math.max.apply(null, kde);
    const scale = (ui.getDistributionHeight() - 2) / kdeMax;

    const ctx = ui.distributionCanvas.getContext('2d');
    ctx.clearRect(0, 0, ui.getDistributionWidth(), ui.getDistributionHeight());
    ctx.beginPath();
    ctx.moveTo(0, (ui.getDistributionHeight() - 1) - (kde[0] * scale));
    for (let x = 1; x < kde.length; ++x)
        ctx.lineTo(x, (ui.getDistributionHeight() - 1) - (kde[x] * scale));
    ctx.strokeStyle = 'rgb(171, 181, 202)'; // #abb5ca
    ctx.lineWidth = 2;
    ctx.stroke();

    const maxValue = Math.max.apply(null, result.cellData.cellSizesSqrt) + 1;
    const step = maxValue / kde.length;

    for (const sizeThresh of sizeRange) {
        if (sizeThresh >= 0) {
            //console.log('sizeThresh:', sizeThresh, ui.sizeThreshSlider1.value);
            const pos = Math.sqrt(sizeThresh) / step;
            ctx.beginPath();
            ctx.moveTo(pos, 0);
            ctx.lineTo(pos, ui.getDistributionHeight());
            ctx.strokeStyle = 'rgb(255, 255, 255)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }

    if (view.selectedCellSize > 0) {
        const pos = Math.sqrt(view.selectedCellSize) / step;
        ctx.beginPath();
        ctx.moveTo(pos, 0);
        ctx.lineTo(pos, ui.getDistributionHeight());
        ctx.strokeStyle = 'rgb(255, 255, 0)';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}


function drawMarkerHistogram()
{
    const hist = result.markerHistogram;
    const histMax = Math.max.apply(null, hist);

    ui.histogramCanvas.width = hist.length;
    ui.histogramCanvas.height = 1000;
    const scale = 1000.0 / histMax;

    const ctx = ui.histogramCanvas.getContext('2d');
    ctx.clearRect(0, 0, ui.getHistogramWidth(), ui.getHistogramHeight());
    ctx.fillStyle = 'rgb(171, 181, 202)'; // #abb5ca
    for (let i = 0; i < hist.length; ++i)
        ctx.fillRect(i, 1000, 1, -hist[i]*scale);
}


function updateScore()
{
    document.getElementById('scoring-num-total').innerHTML = result.score.countTotal;
    document.getElementById('scoring-num-pos').innerHTML = result.score.countPos;
    document.getElementById('scoring-percentage').innerHTML = result.score.percentPosStr;
}


function updateMarkerTicks(range)
{
    document.getElementById('markerMinTick').style.left = (range[0] / 255 * 100) + '%';
    document.getElementById('markerMaxTick').style.right = (100 - (range[1] / 255 * 100)) + '%';
}


function createMultiplex(settings)
{
    //updateMarkerTicks();

    const images = [];
    if (settings.showMarker) {
        let colorMarker = colorizeChannel(pixelData.marker, 255, 255, 0, settings.markerRange);
        images.push(colorMarker);
    }
    if (settings.showDapi) {
        let colorDapi = colorizeChannel(pixelData.dapi, 0, 0, 255, settings.dapiRange);
        images.push(colorDapi);
    }
    if (settings.showLap2) {
        let colorLap2 = colorizeChannel(pixelData.lap2, 0, 255, 0, settings.lap2Range);
        images.push(colorLap2);
    }

    const multiplex = new ImageData(pixelData.width, pixelData.height);
    for (let i = 0, j = 0; i < multiplex.data.length; i += 4, j += 3) {
        let r = 0;
        let g = 0;
        let b = 0;
        for (let n = 0; n < images.length; ++n) {
            r += images[n][j];
            g += images[n][j+1];
            b += images[n][j+2];
        }
        multiplex.data[i] = r;
        multiplex.data[i+1] = g;
        multiplex.data[i+2] = b;
        multiplex.data[i+3] = 255;
    }

    return multiplex;
}


function colorizeChannel(channel, r, g, b, range = [0, 255])
{
    let multiplier = 255 / (range[1] - range[0]);
    let colorized = new Uint8Array(channel.length * 3);
    r = r / 255;
    g = g / 255;
    b = b / 255;
    for (let i = 0, j = 0; i < channel.length; ++i, j += 3) {
        let val = (channel[i] - range[0]) * multiplier;
        val = (val < 0) ? 0 : ((val > 255) ? 255 : val);
        colorized[j] = val * r;
        colorized[j+1] = val * g;
        colorized[j+2] = val * b;
    }
    return colorized;
}


function mapCanvasToImageCoord(offsetX, offsetY)
{
    let imageX = Math.round((1/pz.zoom) * offsetX - pz.shiftX);
    if (imageX < 0)
        imageX = 0;
    else if (imageX >= pixelData.width)
        imageX = pixelData.width - 1;

    let imageY = Math.round((1/pz.zoom) * offsetY - pz.shiftY);
    if (imageY < 0)
        imageY = 0;
    else if (imageY >= pixelData.height)
        imageY = pixelData.height - 1;

    return {x: imageX, y: imageY};
}


function selectCellSize(offsetX, offsetY)
{
    //console.log(offsetX, offsetY);
    const {x, y} = mapCanvasToImageCoord(offsetX, offsetY);
    //console.log(x, y);

    const idx = y * pixelData.width + x;
    const value = result.posneg[idx];
    //console.log('idx: ', idx, '  posneg[idx]: ', value);

    view.selectedCellSize = 0;

    if (value == 2 || value == -2)
    {
        const width = pixelData.width;
        const neigh_x = [-1, 0, 1, -1, 1, -1, 0, 1];
        const neigh_y = [-1, -1, -1, 0, 0, 1, 1, 1];
        const neigh_i = [-width-1, -width, -width+1, -1, 1, width-1, width, width+1];
        let cluster = [idx];
        let pixels_i = [idx];
        result.posneg[idx] = 50;

        while (pixels_i.length > 0)
        {
            let i = pixels_i.pop();
            let px = i % width;
            let py = Math.floor(i / width);
            for (let n = 0; n < neigh_i.length; ++n)
            {
                let nx = px + neigh_x[n];
                let ny = py + neigh_y[n];
                let ni = i + neigh_i[n];
                if (nx < 0 || nx >= pixelData.width || ny < 0 || ny >= pixelData.height)
                    continue;
                if (result.posneg[ni] == value) {
                    cluster.push(ni);
                    pixels_i.push(ni);
                    result.posneg[ni] = 50;
                }
            }
        }

        for (const index of cluster)
            result.posneg[index] = value;

        if (cluster.length > result.settings.sizeRange[0] && cluster.length < result.settings.sizeRange[1]) {
            view.selectedCellSize = cluster.length;
            //console.log('cell size: ', view.selectedCellSize);
        }
    }

    updateResults();
}


function resetMask(e)
{
    e.preventDefault();
    roi.deleteAllRegions();
}


function saveState()
{
    if (!result.processing)
    {
        //console.log('SAVE STATE', needToSaveState);

        const defaults = {};
        //defaults['segThresh'] = ui.defaultSegThresh;
        defaults['sizeRange'] = [Math.round(ui.defaultSizeThresh**2), Math.round(ui.defaultSizeMax**2)];
        /* For old mIF outputs
        defaults['markerThresh'] = ui.defaultMarkerThresh;
        defaults['mpMarkerRange'] = [ui.defaultMarkerRange[0], ui.defaultMarkerRange[1]];
        defaults['mpDapiRange'] = [ui.defaultDapiRange[0], ui.defaultDapiRange[1]];
        defaults['mpLap2Range'] = [ui.defaultLap2Range[0], ui.defaultLap2Range[1]];
        */

        const settings = {};
        //settings['segThresh'] = result.settings.segThresh;
        settings['sizeRange'] = result.settings.sizeRange;
        settings['sizeMax'] = Math.round(ui.sizeThreshSlider1.max**2);
        settings['markerThresh'] = result.settings.markerThresh;
        //settings['view'] = result.settings.originalView ? 'original' : 'multiplex';
        settings['showSeg'] = result.settings.showOverlay;
        /* For old mIF outputs
        settings['mpMarkerVisible'] = result.settings.showMarker;
        settings['mpMarkerRange'] = result.settings.markerRange;
        settings['mpDapiVisible'] = result.settings.showDapi;
        settings['mpDapiRange'] = result.settings.dapiRange;
        settings['mpLap2Visible'] = result.settings.showLap2;
        settings['mpLap2Range'] = result.settings.lap2Range;
        */

        const scores = {};
        scores['numTotal'] = result.score.countTotal;
        scores['numPos'] = result.score.countPos;

        const rois = roi.saveRegions();

        const state = {timestamp: Date.now()};
        state['defaults'] = defaults;
        state['settings'] = settings;
        state['scores'] = scores;
        state['rois'] = rois;
        state['dataVersion'] = 2;
        //console.log('state:', JSON.stringify(state));

        const xhr = new XMLHttpRequest();
        xhr.onload = () => {
            if (xhr.status === 200) {
                //console.log(xhr.responseText);
            } else {
                console.log(xhr.responseText);
            }
        };
        xhr.open('POST', '/result/state/' + imageID);
        xhr.send(JSON.stringify(state));

        needToSaveState = false;
    }
}


function downloadFiles(e)
{
    e.preventDefault();
    //console.log('downloadFiles()');

    ui.downloadFiles.style.display = 'none';
    ui.preparingFiles.style.display = 'inline';
    window.setTimeout(() => prepareFiles(), 100);
}


function prepareFiles()
{
    var zip = new JSZip();

    let scoring = 'Number of total nuclei: ' + result.score.countTotal + '\r\n';
    scoring += 'Number of IHC+ cells: ' + result.score.countPos + '\r\n';
    scoring += 'Percentage of IHC+ cells: ' + result.score.percentPosStr;
    zip.file('scoring.txt', scoring);

    zip.file('Original.png', imageToBlob(imageData.orig));
    zip.file('Seg.png', imageToBlob(imageData.seg));
    /* For old mIF outputs
    zip.file('DAPI.png', imageToBlob(imageData.dapi));
    zip.file('Lap2.png', imageToBlob(imageData.lap2));
    zip.file('Marker.png', imageToBlob(imageData.marker));
    zip.file('Hema.png', imageToBlob(imageData.hema));
    */

    const composite = new ImageData(imageData.orig.width, imageData.orig.height);
    composite.data.set(imageData.orig.data);
    const overlay = result.overlay;
    for (let i = 0, j = 0; i < overlay.length; ++i, j += 4) {
        if (overlay[i] > 0) {
            composite.data[j] = 255;
            composite.data[j+1] = 0;
            composite.data[j+2] = 0;
        } else if (overlay[i] < 0) {
            composite.data[j] = 0;
            composite.data[j+1] = 0;
            composite.data[j+2] = 255;
        }
    }
    zip.file('SegOverlaid.png', imageToBlob(composite));

    zip.file('SegRefined.png', imageToBlob(generateSegRefined()));

    zip.generateAsync({ type: "blob" }).then(function (blob) {
        ui.preparingFiles.style.display = 'none';
        ui.downloadFiles.style.display = 'inline';
        saveAs(blob, "deepliif_results.zip");
    }, function (err) {
        console.log(err);
    });
}


function generateSegRefined()
{
    const width = pixelData.width;
    const height = pixelData.height;
    const posneg = result.posneg;
    const overlay = result.overlay;

    let pix = new Uint8ClampedArray(posneg.length * 4);
    for (let i = 0; i < posneg.length; ++i) {
        if (posneg[i] == 2) {
            pix[i * 4] = 255;
            pix[i * 4 + 3] = 255;
        } else if (posneg[i] == -2) {
            pix[i * 4 + 2] = 255;
            pix[i * 4 + 3] = 255;
        }
    }
    
    // set outlines to green
    for (let i = 0, j = 0; i < overlay.length; ++i, j += 4) {
        if (overlay[i] != 0) {
            pix[j] = 0;
            pix[j + 1] = 255;
            pix[j + 2] = 0;
            pix[j + 3] = 255;
        }
    }

    // remove too small cells (cells w/o outline)
    for (let y = 0, i = 4; y < height; ++y, i += 4)
        for (let x = 1; x < width; ++x, i += 4)
            if (pix[i + 3] == 255 && pix[i + 1] == 0 && pix[i - 1] == 0)
                pix[i + 3] = 0;
    for (let y = 0; y < height; ++y)
        for (let x = width - 2; x >= 0; --x) {
            let i = (y * width + x) * 4;
            if (pix[i + 3] == 255 && pix[i + 1] == 0 && pix[i + 7] == 0)
                pix[i + 3] = 0;
        }
    
    // change color on Marker segmented cells (neg => pos)
    for (let y = 0, i = 4; y < height; ++y, i += 4)
        for (let x = 1; x < width; ++x, i += 4)
            if (pix[i + 3] == 255 && pix[i + 1] == 0 && pix[i - 1] == 255) {
                if (pix[i - 3] == 255) {
                    const color = (overlay[i/4-1] > 0) ? [255, 0, 0] : [0, 0, 255];
                    pix[i] = color[0];
                    pix[i + 1] = color[1];
                    pix[i + 2] = color[2];
                    //pix[i] = overlay[i - 4];
                    //pix[i + 1] = overlay[i - 3];
                    //pix[i + 2] = overlay[i - 2];
                }
                else {
                    pix[i] = pix[i - 4];
                    pix[i + 1] = pix[i - 3];
                    pix[i + 2] = pix[i - 2];
                }
            }
    for (let y = 0; y < height; ++y)
        for (let x = width - 2; x >= 0; --x) {
            let i = (y * width + x) * 4;
            if (pix[i + 3] == 255 && pix[i + 1] == 0 && pix[i + 7] == 255) {
                if (pix[i + 5] == 255) {
                    const color = (overlay[i/4+1] > 0) ? [255, 0, 0] : [0, 0, 255];
                    pix[i] = color[0];
                    pix[i + 1] = color[1];
                    pix[i + 2] = color[2];
                    //pix[i] = overlay[i + 4];
                    //pix[i + 1] = overlay[i + 5];
                    //pix[i + 2] = overlay[i + 6];
                }
                else {
                    pix[i] = pix[i + 4];
                    pix[i + 1] = pix[i + 5];
                    pix[i + 2] = pix[i + 6];
                }
            }
        }
    
    // make background black
    for (let i = 0; i < pix.length; i += 4) {
        if (pix[i + 3] == 0) {
            pix[i] = 0;
            pix[i + 1] = 0;
            pix[i + 2] = 0;
            pix[i + 3] = 255;
        }
    }

    return new ImageData(pix, width, height);
}