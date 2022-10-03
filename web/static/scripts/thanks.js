$("document").ready(function() {
    if(localStorage.getItem("token") === null){
        window.location = window.location.protocol + "//" + window.location.host + "/";
    }
});

$("#btn-reset").on("click", function(){
    $.ajax({
      url: "reset",
      headers: { 'x-access-tokens': localStorage.getItem("token") }
    });
    $.ajax({
      url: "reset",
      headers: { 'x-access-tokens': localStorage.getItem("oldToken") }
    }).done(() => {window.location = window.location.protocol + "//" + window.location.host + "/";});
})