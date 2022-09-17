$("#btn-proceed").on("click",function() {
    $.ajax({
      url: "startRun"
    }).done( data => {
          localStorage.setItem("token", data.id);
          window.location = window.location = window.location.protocol + "//" + window.location.host + "/" + "feedback";
        }
    );
})