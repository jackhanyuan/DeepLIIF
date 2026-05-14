var theName = '';
var theWidth = 0;
var theHeight = 0;
var isSample = false;


window.addEventListener('load', initDragDrop);


function hide(elementName)
{
    let element = document.getElementById(elementName);
    element.classList.remove('show');
    element.classList.add('hide');
}


function show(elementName)
{
    let element = document.getElementById(elementName);
    element.classList.remove('hide');
    element.classList.add('show');
}


function showError(message)
{
    document.getElementById('error-message').innerHTML = message;
    hide('processing');
    show('error');
}


function initDragDrop()
{
	var dropArea = document.getElementById("drop-area");
	dropArea.addEventListener("dragover", e =>{
		e.preventDefault();
		dropArea.style.backgroundColor = "#0c1e2b";
	});
	["dragleave", "dragend"].forEach(type => {
		dropArea.addEventListener(type, e=>{
			dropArea.style.backgroundColor = "#07090C";
		});
	});
	dropArea.addEventListener("drop", e=>
	{
		e.preventDefault();
		dropArea.style.backgroundColor = "#07090C";
        let uriList = e.dataTransfer.getData('text/uri-list');
        if (uriList != '') {
            let parts = uriList.split('#');
            if (parts.length != 2)
                return;
            let name = parts[1];
            if (name in sampleExtensions)
                sample(name, sampleExtensions[name], sampleResolutions[name]);
        }
        else if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
	});
}


function chooseFile()
{
	let input = document.createElement('input');
	input.type = 'file';
	input.onchange = _ => {
	    let file = Array.from(input.files);
        uploadFile(file[0]);
	};
	input.click();
}


function uploadFile(file)
{
    videoMimes = [
        'video/mp4',
        'video/avi',
        'video/msvideo',
        'video/x-msvideo',
    ];
    videoExts = [
        'mp4',
        'avi',
        'mov',
    ];

    if (videoMimes.includes(file.type))
        uploadVideo(file);
    else if (videoExts.includes(file.name.split('.').pop().toLowerCase()))
        uploadVideo(file);
    else
        uploadImage(file);
}


function uploadImage(file)
{
    showTab('samples');
    hide('upload-start');
    //hide('samples-section');
    hide('samples');
    show('processing');

    if (file.size > 128 * 1024 * 1024) {
        showError("The file size is too large.");
        return;
    }

    var formData = new FormData();
    formData.append('image', file);
    if (isSample)
        formData.append('label', 'sample');

    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                uploadDoneImage(xhr.responseText);
            } else {
                showError(xhr.responseText);
            }
        }
    };
    xhr.open('POST', '/upload-image');
    xhr.send(formData);
}


function uploadVideo(file)
{
    document.getElementById('processing-text').innerHTML = 'Uploading Video';
    showTab('faq');
    hide('upload-start');
    //hide('samples-section');
    hide('samples');
    show('processing');
    show('progress-box');
    document.getElementById('progress-box').style.visibility = 'visible';

    if (file.size > 3072 * 1024 * 1024) {
        showError("The file size is too large.");
        return;
    }

    var formData = new FormData();
    formData.append('video', file);
    formData.append('ext', 'mp4'); // set this to original file extension?

    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                uploadDoneVideo(xhr.responseText);
            } else {
                showError(xhr.responseText);
            }
        }
    };
    xhr.upload.addEventListener('progress', (e) => {
        var percentUploaded = 100 * e.loaded / e.total;
        document.getElementById('progress-bar').style.right = (100 - percentUploaded) + '%';
        if (e.loaded == e.total)
            document.getElementById('processing-text').innerHTML = 'Finalizing Upload';
    });
    xhr.open('POST', '/upload-video');
    xhr.send(formData);
}


function uploadDoneImage(response)
{
    let json = JSON.parse(response);
    theName = json.name;

    theWidth = json.width;
    theHeight = json.height;
    updateTimeEstimate();

    let img = document.createElement("img");
    img.setAttribute('src', json.thumb);
    let original = document.getElementsByClassName("uploaded-image");
    original[0].appendChild(img);

    hide('processing');
    show('upload-done');
}


function uploadDoneVideo(response)
{
    let json = JSON.parse(response);
    theName = json.name;

    //theWidth = json.width;
    //theHeight = json.height;
    //updateTimeEstimate();

    let img = document.createElement("img");
    img.setAttribute('src', json.thumb);
    thumbnail = document.getElementById('uploaded-video-thumb');
    thumbnail.appendChild(img);

    document.getElementById('progress-box').style.visibility = 'hidden';
    hide('progress-box');
    hide('processing');
    show('upload-done-video');
}


let countStatusTries = 0;
let inferStartTime = 0, stitchStartTime = 0;
let updateProgressBarIntervalID = 0;


function inferStart()
{
    document.getElementById('processing-text').innerHTML = 'DeepLIIF Pending';
    hide('upload-done');
    show('progress-box');
    show('processing');
    countStatusTries = 0;

    const resolution = document.getElementById('upload-resolution');
    const params = `?name=${theName}&resolution=${resolution.value}`;

    const xhr = new XMLHttpRequest();
    xhr.onload = () => {
        if (xhr.status === 200) {
            var json = JSON.parse(xhr.responseText);
            theName = json.name;
            inferStartTime = 0;
            inferStatus();
        } else {
            showError(xhr.responseText);
        }
    };
    xhr.open('POST', '/infer-start'+params);
    xhr.send();
}


async function inferStatus()
{
    ++countStatusTries;
    let waitTime = 1000;
    if (countStatusTries > 24)
        waitTime = 6000;
    else if (countStatusTries > 22)
        waitTime = 5000;
    else if (countStatusTries > 19)
        waitTime = 4000;
    else if (countStatusTries > 15)
        waitTime = 3000;
    else if (countStatusTries > 10)
        waitTime = 2000;

    await new Promise(r => setTimeout(r, waitTime));

    const params = `?name=${theName}`;

    const xhr = new XMLHttpRequest();
    xhr.onload = () => {
        if (xhr.status === 200) {
            let state = JSON.parse(xhr.responseText).state;
            if (state == 'SUCCESS') {
                clearInterval(updateProgressBarIntervalID);
                if (inferStartTime == 0) {
                    document.getElementById('processing-text').innerHTML = 'DeepLIIF Running';
                    document.getElementById('progress-box').style.visibility = 'visible';
                }
                document.getElementById('progress-bar').style.right = '1px';
                inferDone(xhr.responseText);
            } else if (state == 'FAILURE' || state == 'REVOKED') {
                clearInterval(updateProgressBarIntervalID);
                showError('Error in processing image.');
            } else {
                if (state == 'STARTED' && inferStartTime == 0) {
                    inferStartTime = new Date();
                    document.getElementById('processing-text').innerHTML = 'DeepLIIF Running';
                    document.getElementById('progress-box').style.visibility = 'visible';
                    updateProgressBarIntervalID = setInterval(updateProgressBar, 1000);
                }
                inferStatus();
            }
        } else {
            if (xhr.status == 0)
                inferStatus();
            else
                showError(xhr.responseText);
        }
    };
    xhr.open('POST', '/infer-status'+params);
    xhr.send();
}


function inferDone(response)
{
    var json = JSON.parse(response);
    theName = json.name;
    window.location.href = '/result/' + theName;
}


function stitchStart()
{
    document.getElementById('processing-text').innerHTML = 'DeepLIIF Stitch Pending';
    hide('upload-done-video');
    show('progress-box');
    show('processing');
    countStatusTries = 0;

    const resolution = document.getElementById('upload-resolution-video');
    const params = `?name=${theName}&resolution=${resolution.value}`;

    const xhr = new XMLHttpRequest();
    xhr.onload = () => {
        if (xhr.status === 200) {
            var json = JSON.parse(xhr.responseText);
            theName = json.name;
            stitchStartTime = 0;
            window.location.href = '/queue/' + theName;
        } else {
            showError(xhr.responseText);
        }
    };
    xhr.open('POST', '/stitch-start'+params);
    xhr.send();
}


function updateProgressBar()
{
    if (inferStartTime != 0) {
        const elapsedTime = Math.round((new Date() - inferStartTime) / 1000)
        let percentElapsed1 = (elapsedTime * 100) / estimatedInferTime;
        let percentElapsed2 = (elapsedTime * 100) / (2 * estimatedInferTime);
        let percentElapsed3 = (elapsedTime * 100) / (3 * estimatedInferTime);
        if (percentElapsed1 > 100)
            percentElapsed1 = 100
        if (percentElapsed2 > 100)
            percentElapsed2;
        if (percentElapsed3 > 100)
            percentElapsed3;
        let avgPercentElapsed = (percentElapsed1 * 0.5) + (percentElapsed2 * 0.3) + (percentElapsed3 * 0.2);
        //let avgPercentElapsed = (percentElapsed1 * 0.34) + (percentElapsed2 * 0.33) + (percentElapsed3 * 0.33);
        if (avgPercentElapsed > 98)
            avgPercentElapsed = 98
        document.getElementById('progress-bar').style.right = (100 - avgPercentElapsed) + '%';
    }
}


function sample(filename, extension, resolution)
{
    hide('upload-start');
    //hide('samples-section');
    hide('samples');
    show('processing');

    isSample = true;
    document.getElementById('upload-resolution').value = parseInt(resolution);
    fetch('/static/samples3/' + filename + '.' + extension)
        .then(response => response.blob())
        .then(blob => {
            var file = new File([blob], filename+'.'+extension, {type: 'image/'+extension});
            uploadImage(file);
        });
}


function acknowledgeTerms()
{
    if (document.getElementById('terms-check').checked) {
        document.getElementById('please-read').style.display = 'none';
        document.getElementById('accept-terms').style.display = 'inline';
    }
    else {
        document.getElementById('accept-terms').style.display = 'none';
        document.getElementById('please-read').style.display = 'block';
    }
}


function acceptTerms()
{
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/accept', false);
    xhr.send(null);
}


let estimatedInferTime = 0; // in seconds

function updateTimeEstimate()
{
    let size = theWidth * theHeight;
    let resolution = document.getElementById('upload-resolution').value;
    let time = 'Unknown';
    if (resolution == '5') {
        if (size <= 262144) {
            time = 'A few seconds';
            estimatedInferTime = 15;
        } else if (size <= 655360) {
            time = '30 seconds';
            estimatedInferTime = 30;
        } else if (size <= 1376256) {
            time = '1 minute';
            estimatedInferTime = 60;
        } else if (size <= 2621440) {
            time = '2 minutes';
            estimatedInferTime = 120;
        } else if (size <= 3670016) {
            time = '3 minutes';
            estimatedInferTime = 180;
        } else if (size <= 4915200) {
            time = '4 minutes';
            estimatedInferTime = 240;
        } else if (size <= 5898240) {
            time = '5 minutes';
            estimatedInferTime = 300;
        } else if (size <= 7340032) {
            time = '6 minutes';
            estimatedInferTime = 360;
        } else if (size <= 8388608) {
            time = '7 minutes';
            estimatedInferTime = 420;
        } else if (size <= 9437184) {
            time = '8 minutes';
            estimatedInferTime = 480;
        } else if (size <= 10485760) {
            time = '9 minutes';
            estimatedInferTime = 540;
        } else {
            time = '> 10 minutes';
            estimatedInferTime = 600;
        }
    } else if (resolution == '10') {
        if (size <= 458752) {
            time = 'A few seconds';
            estimatedInferTime = 15;
        } else if (size <= 1441792) {
            time = '30 seconds';
            estimatedInferTime = 30;
        } else if (size <= 2949120) {
            time = '1 minute';
            estimatedInferTime = 60;
        } else if (size <= 5505024) {
            time = '2 minutes';
            estimatedInferTime = 120;
        } else if (size <= 7929856) {
            time = '3 minutes';
            estimatedInferTime = 180;
        } else if (size <= 10485760) {
            time = '4 minutes';
            estimatedInferTime = 240;
        } else if (size <= 12845056) {
            time = '5 minutes';
            estimatedInferTime = 300;
        } else if (size <= 14745600) {
            time = '6 minutes';
            estimatedInferTime = 360;
        } else {
            time = '7 minutes';
            estimatedInferTime = 420;
        }
    } else if (resolution == '20') {
        if (size <= 655360) {
            time = 'A few seconds';
            estimatedInferTime = 15;
        } else if (size <= 2293760) {
            time = '30 seconds';
            estimatedInferTime = 30;
        } else if (size <= 5046272) {
            time = '1 minute';
            estimatedInferTime = 60;
        } else if (size <= 8650752) {
            time = '2 minutes';
            estimatedInferTime = 120;
        } else if (size <= 12582912) {
            time = '3 minutes';
            estimatedInferTime = 180;
        } else {
            time = '4 minutes';
            estimatedInferTime = 240;
        }
    } else if (resolution == '40') {
        if (size <= 655360) {
            time = 'A few seconds';
            estimatedInferTime = 15;
        } else if (size <= 2883584) {
            time = '30 seconds';
            estimatedInferTime = 30;
        } else if (size <= 6815744) {
            time = '1 minute';
            estimatedInferTime = 60;
        } else if (size <= 11534336) {
            time = '2 minutes';
            estimatedInferTime = 120;
        } else {
            time = '3 minutes';
            estimatedInferTime = 180;
        }
    }
    document.getElementById('est-time').innerHTML = time;
}


function replaceButton() {
    document.getElementById('replaceBtn').style.display = 'none';
    document.getElementById('hiddenContent').style.display = 'block';
}

function replaceButtonStitch() {
    document.getElementById('replaceBtnStitch').style.display = 'none';
    document.getElementById('hiddenContentStitch').style.display = 'block';
}

function toggleStitchSection(name) {
    document.getElementById(`stitch-${name}-button`).classList.toggle('open');
    const content = document.getElementById(`stitch-${name}-content`);
    if (content.style.maxHeight)
        content.style.maxHeight = null;
    else
        content.style.maxHeight = `${content.scrollHeight}px`;
}


// Helper to position the triangle under the active tab
function positionTriangle() {
    const container = document.getElementById('tabs-container');
    const leftTab = document.getElementById('tab-samples');
    const rightTab = document.getElementById('tab-faq');
    let activeTab = leftTab.classList.contains('active') ? leftTab : rightTab;
    let rect = activeTab.getBoundingClientRect();
    let containerRect = container.getBoundingClientRect();
    // For left tab: align to bottom right, shift left 50px
    // For right tab: align to bottom left, shift right 50px
    let x = 0;
    if (activeTab === leftTab) {
        x = rect.right - containerRect.left - 50 - 12; // 12 is half triangle width
    } else {
        x = rect.left - containerRect.left + 50 - 12;
    }
    container.style.setProperty('--triangle-x', x + 'px');
}


function showTab(tab) {
    const tabSamples = document.getElementById('tab-samples');
    const tabFaq = document.getElementById('tab-faq');
    const samplesSection = document.getElementById('samples-section');
    const faqSection = document.getElementById('faq-section');

    if (tab === 'samples') {
        tabSamples.classList.add('active');
        tabFaq.classList.remove('active');
        samplesSection.classList.add('show');
        faqSection.classList.remove('show');
    } else {
        tabSamples.classList.remove('active');
        tabFaq.classList.add('active');
        samplesSection.classList.remove('show');
        faqSection.classList.add('show');
    }
    positionTriangle();
}


// Initial triangle position on load and on resize
window.addEventListener('DOMContentLoaded', positionTriangle);
window.addEventListener('resize', positionTriangle);
