chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        console.log("Entered background " + request.message + " " + request.type + " " + request.content);
		if (request.message == "listeners" && request.type == "background") {
			$.ajax({
				url: "http://127.0.0.1:5000/_get_data/",
				type: "POST",
				data: { arg1: request.content} ,
				success: function(resp){
				  console.log(resp);
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
