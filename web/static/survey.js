$("document").ready(function() {
    if(localStorage.getItem("token") === null){
        window.location = window.location.protocol + "//" + window.location.host + "/";
    }
});

$("#btn-undo").on("click", function(){
    $.ajax({
      url: "restart",
      headers: { 'x-access-tokens': localStorage.getItem("token") }
    }).done( data => {
          localStorage.setItem("token", data.id);
          localStorage.getItem("token")
          window.location = window.location.protocol + "//" + window.location.host + "/feedback";
        }
    );
})

$("#btn-submit").on("click", function(){
    window.location = window.location.protocol + "//" + window.location.host + "/thanks";
})