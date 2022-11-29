const image_input = document.querySelector("#image-input");

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

    let response = await fetch('http://127.0.0.1:5000/image', {
      method: 'POST',
      body: new FormData(formElem)
    });
     const imageBlob = await response.blob();
     const imageObjectURL = URL.createObjectURL(imageBlob);
     const rImage = document.createElement('img');
     rImage.src = imageObjectURL;
     rImage.style.width = "256px";
     rImage.style.height = "256px";

     const container = document.getElementById('receive-image');
     while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
     container.appendChild(rImage);
    
  };