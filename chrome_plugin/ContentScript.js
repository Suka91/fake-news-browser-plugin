window.onload = function(){ 
    console.log("DBG: Loaded.");
    console.log(window.location.href);
    chrome.runtime.sendMessage({message: "listeners", type: "background", content: window.location.href}, function(response) {
        });
};
