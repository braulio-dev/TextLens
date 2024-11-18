import { ref } from 'vue'

export const fileInput = ref(null)
export const isLoading = ref(false)
export const lang = ref('spa')

export const handleFileChange = (event, emit) => {
  const file = event.target.files[0]
  if (file) {
    emit('block-summary')
    processFile(file, emit)
  }
}

export const processFile = (file, emit) => {
  const formData = new FormData()
  formData.append('image', file)

  isLoading.value = true
  fetch('http://localhost:8000/summarize/?lang=' + lang.value, {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    emit('summary-received', data.summary, data.extracted_text)
  })
  .catch(error => console.error(error))
  .finally(() => {
    isLoading.value = false
  })
}

export const triggerFileInput = () => {
  fileInput.value.click()
}

export const handlePasteText = async (emit) => {
  try {
    const text = await navigator.clipboard.readText()
    if (text) {
      emit('block-summary')
      processText(text, emit)
    }
  } catch (error) {
    console.error('Failed to read clipboard contents: ', error)
  }
}

export const processText = (text, emit) => {
  const payload = { text, lang: lang.value }

  isLoading.value = true
  fetch('http://localhost:8000/summarize-text/?lang=' + lang.value, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  })
  .then(response => response.json())
  .then(data => {
    emit('summary-received', data.summary)
  })
  .catch(error => console.error(error))
  .finally(() => {
    isLoading.value = false
  })
}

export const handlePasteImage = async (emit) => {
  try {
    const items = await navigator.clipboard.read()
    for (const item of items) {
      if (item.types.includes('image/png')) {
        const blob = await item.getType('image/png')
        emit('block-summary')
        processFile(blob, emit)
      }
    }
  } catch (error) {
    console.error('Failed to read clipboard contents: ', error)
  }
}
