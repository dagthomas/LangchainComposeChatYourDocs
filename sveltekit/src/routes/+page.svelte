<script type="ts">
	let types = 'Files';
	import { enhance } from '$app/forms';
	import { goto } from '$app/navigation';
	import { fade, fly } from 'svelte/transition';
	export let form;
	let formData;

	$: if (form?.result) {
		if (form?.result === 'Filetype not supported') {
			alert('Filetype not supported');
			sending = false;
		} else {
			let returnData = form?.result;
			let returnString = `/collection/${returnData}`;
			goto(returnString);
		}
	}
	var num_links = 1;
	const addField = () => {
		if (num_links < 5) {
			num_links += 1;
		}
	};

	const removeField = (div) => {
		num_links -= 1;
	};
	let sending = false;
	let collection_name = `collection-${Math.floor(Math.random() * 100000)}`;
</script>

<div class="flex flex-col items-center justify-center h-screen">
	{#if !sending}
		<div class="flex items-center justify-center flex-col w-2/6">
			<div class="pr-8" style="transform: rotate(90deg);">
				<div class="form-control">
					<label class="label cursor-pointer">
						<span class="label-text pr-4" style="transform: rotate(-12deg);">Webpages</span>
						<input
							bind:group={types}
							value={'Webpages'}
							type="radio"
							name="radio-10"
							class="radio checked:bg-secondary"
							checked
						/>
					</label>
				</div>
				<div class="form-control">
					<label class="label cursor-pointer">
						<span class="label-text pr-4 " style="transform: rotate(-12deg);">Webpage</span>
						<input
							bind:group={types}
							value={'Webpage'}
							type="radio"
							name="radio-10"
							class="radio checked:bg-blue-500"
							checked
						/>
					</label>
				</div>
				<div class="form-control">
					<label class="label cursor-pointer">
						<span class="label-text pr-4" style="transform: rotate(-12deg);">Files</span>
						<input
							bind:group={types}
							value={'Files'}
							type="radio"
							name="radio-10"
							class="radio checked:bg-primary-500"
							checked
						/>
					</label>
				</div>
			</div>
			<div class="w-full">
				{#if types === 'Files'}
					<div in:fade>
						<h1 class="mb-1 font-bold text-lg">Upload a file</h1>
						<p class="mb-4">.PDF, .CSV, .EPUB, .PPTX, .DOCX, .XLSX, .SRT</p>
						<form
							action="?/upload"
							method="POST"
							enctype="multipart/form-data"
							use:enhance
							bind:this={formData}
						>
							<input
								type="file"
								name="file"
								accept=".pdf, .csv, .epub, .pptx, .docx, .xlsx, .xls, .srt"
								class="file-input w-full max-w-full mr-4 border-ll-s-1 border-2 "
								on:change={() => {
									formData.requestSubmit();
									sending = true;
								}}
							/>
						</form>
					</div>
				{:else if types === 'Webpage'}
					<div in:fade>
						<h1 class="mb-4 font-bold text-lg ">Enter a webpage</h1>
						<p class="mb-4">Enter URL to index body text</p>
						<form
							action="?/webpage"
							method="POST"
							bind:this={formData}
							enctype="multipart/form-data"
							use:enhance
						>
							<input
								name="link"
								type="text"
								placeholder="URL"
								class="input  w-full max-w-full mr-4 border-ll-s-1 border-2"
								bind:this={formData}
							/>
							<button
								class="btn mt-4"
								on:click={() => {
									sending = true;
									formData.requestSubmit();
								}}>SEND</button
							>
							{sending}
						</form>
					</div>
				{:else if types === 'Webpages'}
					<div in:fade>
						<h1 class="mb-4 font-bold text-lg ">Enter multiple webpages</h1>
						<p class="mb-4">Enter URLs to index body text</p>
						<form
							action="?/webpages"
							method="POST"
							bind:this={formData}
							enctype="multipart/form-data"
							use:enhance
						>
							<input
								name="collection_name"
								bind:value={collection_name}
								type="text"
								placeholder="URL"
								class="input input-sm  w-full max-w-full mr-4 border-ll-s-1 border-2 mb-4"
							/>
							{#each Array(num_links) as _, i}
								<input
									id={`link_${i}`}
									name={`link_${i}`}
									type="text"
									placeholder="URL"
									class="input   w-full max-w-full mr-4 border-ll-s-1 border-2 mb-4"
								/>
							{/each}
							{#if num_links > 1}
								<button on:click|preventDefault={removeField}>[- remove last URL]</button>
							{/if}
							{#if num_links < 5}
								<button on:click|preventDefault={addField}>[+ add a URL]</button>
							{/if}<br />
							<button
								class="btn mt-4"
								on:click={() => {
									sending = true;
									formData.requestSubmit();
								}}>SEND</button
							>
						</form>
					</div>
				{/if}
			</div>
		</div>
	{:else}
		<div
			role="status"
			class="absolute -translate-x-1/2 -translate-y-1/2 top-2/4 left-1/2"
			in:fade={{ duration: 550 }}
			out:fade
		>
			<svg
				aria-hidden="true"
				class="w-24 h-24 mr-2 animate-spin text-gray-600 fill-ll-s-1-signal"
				viewBox="0 0 100 101"
				fill="none"
				xmlns="http://www.w3.org/2000/svg"
				><path
					d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
					fill="currentColor"
				/><path
					d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
					fill="currentFill"
				/></svg
			>
			<span class="sr-only">Loading...</span>
		</div>
	{/if}
</div>
