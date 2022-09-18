$("document").ready(function() {
    if(localStorage.getItem("token") === null){
        window.location = window.location.protocol + "//" + window.location.host + "/";
    }
});