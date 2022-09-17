$("#btn-undo").on("click", function(){
    console.log(localStorage.getItem("token"))
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