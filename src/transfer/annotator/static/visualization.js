let svg = d3.select("svg");

const DATASET_NAME = "elections";

const A4 = {width: 210, height: 297};
const a4Ratio = A4.height / A4.width;

const margin = {top: 20, right: 20, bottom: 20, left: 20};
const header_size = document.querySelector("body > nav").clientHeight;
const toolbar_size = document.querySelector("#main-toolbar").clientHeight;

let width = window.innerWidth - margin.left - margin.right;
let height = window.innerHeight - header_size - toolbar_size;

let leftCol = document.querySelector("#col1")
let leftImage = leftCol.querySelector("img")
let rightCol = document.querySelector("#col2")
let rightImage = rightCol.querySelector("img")

let button_same = document.querySelector("#same")
let button_different = document.querySelector("#different")
let button_interesting = document.querySelector("#interesting")
let button_next = document.querySelector("#next")
let button_reject = document.querySelector("#reject")

if (same === true) {
    button_same.classList.add("active")
} else if (same === false) {
    button_different.classList.add("active")
}

const mainHeight = height - margin.top - margin.bottom;
const a4Width = mainHeight / a4Ratio;

leftCol.style.width = a4Width + "px";
leftCol.style.height = mainHeight + "px";
rightCol.style.width = a4Width + "px";
rightCol.style.height = mainHeight + "px";

function pageImagePath(document_id, page) {
    // 0a99e5fafc2e4c95a76ce303_0_pageimage.png
    return "/static/data/" + DATASET_NAME + "/" + document_id + "_" + page + "_pageimage.png";
}

function drawInvoice(location, document_id, page) {
    let selected_img = null;
    if (location === "left"){
        selected_img = leftImage
    } else if (location === "right"){
        selected_img = rightImage
    } else {
        throw "Unknown location keyword"
    }

    selected_img.src = pageImagePath(document_id, page)
}

drawInvoice("left", docid1, page1)
drawInvoice("right", docid2, page2)

document.addEventListener('keydown', keyDownHandler, false);
function keyDownHandler(event) {
    console.log(event.keyCode)
    if(event.keyCode === 83) {
        // s
        button_same.click();
        document.activeElement.blur();
    }
    else if(event.keyCode === 68) {
        // d
        button_different.click();
        document.activeElement.blur();
    }
    else if(event.keyCode === 73) {
        // i
        button_interesting.click();
        document.activeElement.blur();
    }
    else if(event.keyCode === 78) {
        // n
        button_next.click();
        document.activeElement.blur();
    }

}
