$("#btn-proceed").on("click",function() {
    $.ajax({
      url: "startRun"
    }).done( data => {
          localStorage.setItem("token", data.id);
          localStorage.setItem("over", false);
          window.location = window.location.protocol + "//" + window.location.host + "/feedback";
        }
    );
})