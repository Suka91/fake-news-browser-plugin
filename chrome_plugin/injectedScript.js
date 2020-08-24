chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        console.log("Entered injectedScript " + request.message + " " + request.type + " " + request.content);
		if (request.type == "listeners" && request.type == "injected_script") {
			console.log(request.content);
		}
		return true;
});

