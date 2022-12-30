const image_input = document.querySelector("#image-input");
const save_button = document.querySelector("#saveImage");
const saveDB_button = document.querySelector("#saveDB");
save_button.disabled = true;
saveDB_button.disabled = true;
var imageObjectURL = null;

image_input.addEventListener("change", function() {
  const reader = new FileReader();
  reader.addEventListener("load", () => {
    const uploaded_image = reader.result;
    document.querySelector("#display-image").style.backgroundImage = `url(${uploaded_image})`;
  });
  reader.readAsDataURL(this.files[0]);
});

formElem.onsubmit = async (e) => {
    e.preventDefault();
    let response = await fetch('/formHandling', {
      method: 'POST',
      body: new FormData(formElem)
    });
    if(response.status==200){
     const imageBlob = await response.blob();
     imageObjectURL = URL.createObjectURL(imageBlob);
     const rImage = document.createElement('img');
     rImage.src = imageObjectURL;
     rImage.style.width = "256px";
     rImage.style.height = "256px";

     const container = document.getElementById('receive-image');
     while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
     container.appendChild(rImage);
     save_button.disabled = false;
     saveDB_button.disabled = false;
     save_button.onclick = download(imageObjectURL);
    }
    else{
      alert("Something wrong with request.");
      console.log(response.text())
  }
  };



  function download(url) {
    var a = document.getElementById("a");
    a.href=url;
    a.download = "face.png";
  }

  async function saveDB(){
    let response = await fetch('/upload_img_to_cloud', {
      method: 'GET',
    });
    if(response.status==200){
      alert("Saved data to DB successfully.");}
    else{
      alert("saveDB went wrong");
    }
  }
