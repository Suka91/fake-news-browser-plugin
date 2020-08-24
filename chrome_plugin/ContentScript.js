console.log("SUKA TEST");
window.onload = function(){ 
    console.log("SUKA TEST 1");
    console.log(window.location.href);
    chrome.runtime.sendMessage({message: "listeners", type: "background", content: window.location.href}, function(response) {
        });
};
