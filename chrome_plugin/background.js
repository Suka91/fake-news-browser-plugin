var serverURL = "http://127.0.0.1:5000"

chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        console.log("Entered background " + request.message + " " + request.type + " " + request.content);
		if (request.message == "listeners" && request.type == "background") {
			chrome.browserAction.setIcon({path: "neutral.png"});
			$.ajax({
				url: serverURL + "/_predict/",
				type: "POST",
				data: { arg1: request.content} ,
				success: function(resp){
				  console.log(resp);
		                  if (resp.data[1] == "1") {
				     chrome.browserAction.setIcon({path: "true.png"});
				  } else {
				     chrome.browserAction.setIcon({path: "false.png"});
				  }
				},
				error: function(e, s, t) {
				  console.log("ERROR OCCURRED");
				  console.log(e);
				  console.log(s);
				  console.log(t);
				}

			});
		    return true;
		}
	}
);
